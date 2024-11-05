import os
from collections import Counter
import numpy as np
import pandas as pd
import torch
import argparse
import tqdm.auto as tqdm
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from multiprocessing import get_context, shared_memory

from sklearn.cluster import KMeans
from sklearn.metrics import (
    rand_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    normalized_mutual_info_score,
)

from .indexed_embeds import IndexedEmbeddings


class LMGC:
    _labels: torch.Tensor = None
    _centroids: torch.Tensor = None
    score = None

    def __init__(
        self,
        n_clusters,
        power=1.0,
        logp_clip=5.0,
        device="cuda",
        max_iter=100,
        tol=1e-6,
        verbose=0,
        phi_estimate_methods="power2",
    ):
        self.n_clusters = n_clusters
        self.power = power
        self.logp_clip = logp_clip
        self.device = device
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.phi_estimate_methods = phi_estimate_methods

    def _prepare(self, logp: np.ndarray):
        logp = torch.tensor(logp, dtype=torch.float32, device=self.device)
        self._clip_(logp, self.logp_clip)

        if self.phi_estimate_methods == "power2":
            power2 = 2.0 * self.power
            logphi = (
                torch.logsumexp(logp * power2, axis=0) - np.log(logp.shape[0])
            ) / power2

        elif self.phi_estimate_methods == "naive":
            logphi = torch.logsumexp(logp, axis=0) - np.log(logp.shape[0])
        else:
            raise ValueError(
                f"Invalid phi estimate method: {self.phi_estimate_methods}"
            )

        ratio = ((logp - logphi) * self.power).exp()
        return logp, ratio

    def fit(self, logp: np.ndarray):
        logp, ratio = self._prepare(logp)
        return self._fit(logp, ratio)

    @property
    def labels(self):
        return self._labels.data.cpu().numpy()

    @property
    def centroids(self):
        return self._centroids.data.cpu().numpy()

    def _fit(self, logp: torch.Tensor, ratio: torch.Tensor):

        # Initialization
        self._centroids = self._random_centroids(logp, ratio)
        self._labels = self._distortion_minimal_assignment(logp, ratio, self._centroids)
        self.score = self._scoring(logp, ratio, self._labels, self._centroids)

        # Clustering
        progress = tqdm.trange(self.max_iter) if self.verbose else range(self.max_iter)
        for _ in progress:

            self._centroids = self._distortion_minimal_centroids(ratio, self._labels)
            self._labels = self._distortion_minimal_assignment(
                logp, ratio, self._centroids
            )

            new_score = self._scoring(logp, ratio, self._labels, self._centroids)
            if self.verbose >= 2:
                print(
                    "Score: {score:g}. Assignments: {assignment}".format(
                        score=new_score,
                        assignment=[
                            c for l, c in Counter(self._labels.tolist()).most_common()
                        ],
                    )
                )
            if new_score > self.score and new_score - self.score < self.tol:
                break

            self.score = new_score

        return self

    def transform(self, logp, ratio):
        batch_size = self._suggested_batch_size(logp)
        n = logp.shape[0]
        distortions = []
        for i in range(0, n, batch_size):
            distortion = self._pairwise_distortion(
                logp[i : i + batch_size].contiguous(),
                ratio[i : i + batch_size].contiguous(),
                self._centroids,
            )
            distortions.append(distortion)
        return torch.cat(distortions, dim=0)

    def fit_transform(self, logp):
        logp, ratio = self._prepare(logp)
        self._fit(logp, ratio)
        return self.transform(logp, ratio)

    def _clip_(self, logp, clip):
        mean_ = logp.mean(0)
        std = logp.std(0)
        idx, idy = torch.where((logp - mean_) / std > clip)
        logp[idx, idy] = mean_[idy] + std[idy] * clip

    def _random_centroids(self, logp, ratio):
        n = logp.shape[0]
        centroids = ratio[np.random.choice(n, self.n_clusters, replace=False)].log()
        centroids -= torch.logsumexp(centroids, dim=-1, keepdim=True)
        return centroids

    def _compute_distortion(self, logp, ratio, centroids):
        """Compute the KL divergence for a set of point-centroid pairs.

        Args:
            logp (Tensor): (N, J)
            ratio (Tensor): (N, J)
            centroids (Tensor): (N, J)

        Returns:
            distortion (Tensor): (N,)
        """
        distortion = (ratio * (logp - centroids)).mean(axis=-1)
        return distortion

    def _pairwise_distortion(self, logp, ratio, centroids):
        """Compute the pairwise KL divergence between a set of points and a set of centroids.

        Args:
            logp (Tensor): (N, J)
            ratio (Tensor): (N, J)
            centroids (Tensor): (C, J)

        Returns:
            distortion (Tensor): (N, C)
        """
        distortion = (ratio[:, None] * (logp[:, None] - centroids)).mean(axis=-1)
        return distortion

    def _scoring(self, logp, ratio, labels, centroids):
        distortions = self._compute_distortion(logp, ratio, centroids[labels])
        score = (-distortions).mean(0)
        return score.item()

    def _suggested_batch_size(self, logp):
        J = logp.size(1)
        batch_size = int(1024**2 * 4 // (self.n_clusters * J))
        return batch_size

    def _distortion_minimal_assignment(self, logp, ratio, centroids):
        labels = []
        batch_size = self._suggested_batch_size(logp)
        n = logp.shape[0]
        for i in range(0, n, batch_size):
            distortions = self._pairwise_distortion(
                logp[i : i + batch_size].contiguous(),
                ratio[i : i + batch_size].contiguous(),
                centroids,
            )
            labels.append(distortions.argmin(axis=-1))
        return torch.cat(labels)

    def _distortion_minimal_centroids(self, ratio, labels, eps=1e-10):
        J = ratio.shape[1]
        centroids = ratio.new_zeros((self.n_clusters, J)) + eps
        centroids.scatter_reduce_(
            0,
            labels[:, None].expand(-1, J).contiguous(),
            ratio,
            reduce="mean",
            include_self=False,
        )
        centroids = centroids.log()
        centroids = centroids - torch.logsumexp(centroids, dim=-1, keepdim=True)
        return centroids


def acc(labels_true, labels_pred):
    n_clusters_true = len(set(labels_true))
    n_clusters_pred = len(set(labels_pred))

    if isinstance(labels_true, list):
        labels_true = np.array(labels_true)

    if isinstance(labels_pred, list):
        labels_pred = np.array(labels_pred)

    matrix = np.zeros((n_clusters_true, n_clusters_pred), dtype=np.int64)
    for i in range(n_clusters_true):
        for j in range(n_clusters_pred):
            matrix[i, j] = np.logical_and(labels_true == i, labels_pred == j).sum()

    row_ind, col_ind = linear_sum_assignment(-matrix)
    return matrix[row_ind, col_ind].sum() / len(labels_true)


def init(device_que):
    os.environ["CUDA_VISIBLE_DEVICES"] = device_que.get()


def run_kmeans(args, X_shm_name, X_shape, X_dtype, n_clusters):
    shm = shared_memory.SharedMemory(name=X_shm_name)
    embs = np.ndarray(X_shape, dtype=X_dtype, buffer=shm.buf)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embs)
    preds = kmeans.labels_
    score = -kmeans.inertia_

    return {
        "labels_pred": preds,
        "score": score,
    }


def run_lmgc(args, X_shm_name, X_shape, X_dtype, n_clusters):
    shm = shared_memory.SharedMemory(name=X_shm_name)
    embs = np.ndarray(X_shape, dtype=X_dtype, buffer=shm.buf)

    lmgc = LMGC(
        n_clusters=n_clusters,
        power=args.power,
        max_iter=100,
        device=f"cuda",
        logp_clip=args.logp_clip,
        verbose=args.verbose,
        phi_estimate_methods=args.phi_estimate_methods,
    )
    if args.J is not None:
        embs = embs[:, : args.J]
    lmgc.fit(embs)
    preds = lmgc.labels
    score = lmgc.score

    return {
        "labels_pred": preds,
        "score": score,
    }


def main(args):
    docs = pd.read_csv(args.docs_path, sep="\t", dtype={"docid": str})
    indexed_embs = IndexedEmbeddings.from_tsv(args.embeddings_path)
    embs = indexed_embs.embs

    def reorder_embs(embs, raw_ids, target_ids):
        N = len(embs)
        assert len(raw_ids) == len(set(raw_ids)) == N
        assert set(raw_ids) == set(target_ids)

        id2pos = {}
        for i, raw_id in enumerate(raw_ids):
            id2pos[raw_id] = i
        mapping = np.zeros(N, dtype=np.int64)
        for i, target_id in enumerate(target_ids):
            mapping[i] = id2pos[target_id]

        return embs[mapping]

    embs = reorder_embs(indexed_embs.embs, raw_ids=indexed_embs.ids, target_ids=docs['docid'].tolist())

    topics = docs["topic"].unique()
    topic2idx = pd.Series(dict(zip(topics, range(len(topics)))))
    targets = topic2idx.loc[docs["topic"]].tolist()

    n_clusters = len(topics)

    ctx = get_context("spawn")
    if args.clustering_method == "kmeans":
        ngpus = 1
    else:
        ngpus = torch.cuda.device_count()

    device_que = ctx.Queue()
    for i in range(ngpus):
        device_que.put(f"{i}")

    shm = shared_memory.SharedMemory(create=True, size=embs.nbytes)
    shared = np.ndarray(embs.shape, dtype=embs.dtype, buffer=shm.buf)
    shared[:] = embs[:]

    with ctx.Pool(ngpus, initializer=init, initargs=(device_que,)) as pool:

        handles = []
        performance = defaultdict(list)
        for _ in tqdm.trange(args.repeats):

            process_args = (args, shm.name, embs.shape, embs.dtype, n_clusters)
            if args.clustering_method == "kmeans":
                handle = pool.apply_async(run_kmeans, args=process_args)
                handles.append(handle)

            elif args.clustering_method == "lmgc":
                handle = pool.apply_async(run_lmgc, args=process_args)
                handles.append(handle)

            else:
                raise ValueError(f"Invalid clustering method: {args.clustering_method}")

        for handle in tqdm.tqdm(handles):
            results_ = handle.get()
            labels_pred = results_["labels_pred"]
            score = results_["score"]

            performance["rand_score"].append(rand_score(targets, labels_pred))
            performance["adjusted_rand_score"].append(
                adjusted_rand_score(targets, labels_pred)
            )
            performance["adjusted_mutual_info_score"].append(
                adjusted_mutual_info_score(targets, labels_pred)
            )
            performance["homogeneity_score"].append(
                homogeneity_score(targets, labels_pred)
            )
            performance["normalized_mutual_info_score"].append(
                normalized_mutual_info_score(targets, labels_pred)
            )
            performance["accuracy"].append(acc(targets, labels_pred))
            performance["score"].append(score)

        performance = pd.DataFrame(performance)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    performance.to_csv(args.output_path, sep="\t")
    print(f"Results saved to {args.output_path}")
    print()

    summary = pd.DataFrame(
        {
            "mean": performance.mean(),
            "best_score": performance.sort_values("score", ascending=False).iloc[0],
            "max": performance.max(),
            "std": performance.std(),
        }
    )
    print(f"----------------- Summary of all {args.repeats} runs -----------------")
    print(summary)
    print(f"----------------- End of summary -----------------")
    print()

    resamples = pd.DataFrame(
        [resampling(performance, args.n_init) for _ in range(args.repeats)]
    )
    summary_of_selecting_procedure = pd.DataFrame(
        {
            "mean": resamples.mean(0),
            "std": resamples.std(0),
        }
    )
    print(f"----------------- Summary of selecting procedure -----------------")
    print(summary_of_selecting_procedure)
    print(f"----------------- End of selecting procedure -----------------")

    shm.close()


def resampling(results, n=10):
    resampled = results.sample(n, replace=True)
    return resampled.sort_values("score", ascending=False).iloc[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_path", type=str)
    parser.add_argument("--embeddings_path", type=str)
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="lmgc",
        choices=["lmgc", "kmeans"],
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--power", type=float, default=0.25)
    parser.add_argument("--logp_clip", type=float, default=5.0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--phi_estimate_methods",
        type=str,
        default="power2",
        choices=["power2", "naive"],
    )
    parser.add_argument("-J", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
