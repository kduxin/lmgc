from typing import List
import os
from collections import Counter
from time import ctime
import numpy as np
import pandas as pd
import argparse
import tqdm.auto as tqdm
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    rand_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    normalized_mutual_info_score,
)

import torch
from sentence_transformers import SentenceTransformer
import transformers

from multiprocessing import get_context, shared_memory


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


@torch.inference_mode()
def call_bert(docs: List[str], args):
    model = transformers.BertModel.from_pretrained(args.embedding_model_name_or_path)
    model.eval()
    model.to("cuda")
    tokenizer = transformers.BertTokenizerFast.from_pretrained(
        args.embedding_model_name_or_path
    )

    batch_size = 512
    embs = []
    for i in tqdm.trange(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        batch = [args.prefix + " " + text for text in batch]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        vecs_batch = outputs.last_hidden_state.mean(1).cpu().numpy()
        embs.append(vecs_batch)
    embs = np.concatenate(embs, axis=0)

    return embs


@torch.inference_mode()
def call_t5_encoder(docs: List[str], args):
    model = transformers.T5EncoderModel.from_pretrained(
        args.embedding_model_name_or_path
    )
    model.eval()
    model.to("cuda")
    tokenizer = transformers.T5TokenizerFast.from_pretrained(
        args.embedding_model_name_or_path
    )

    batch_size = 128
    embs = []
    for i in tqdm.trange(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        batch = [args.prefix + " " + text for text in batch]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        vecs_batch = outputs.last_hidden_state.mean(1).cpu().numpy()
        embs.append(vecs_batch)
    embs = np.concatenate(embs, axis=0)

    return embs


def call_sbert(docs: List[str], args):
    model = SentenceTransformer(args.embedding_model_name_or_path)

    pool = model.start_multi_process_pool()
    embs = model.encode_multi_process(docs, pool, batch_size=256)
    model.stop_multi_process_pool(pool)

    return embs


def embed(docs: List[str], args):
    if args.embedding_method == "bert":
        embs = call_bert(docs, args)
    elif args.embedding_method == "t5-encoder":
        embs = call_t5_encoder(docs, args)
    elif args.embedding_method == "sbert":
        embs = call_sbert(docs, args)
    else:
        raise ValueError(f"Invalid embedding method: {args.embedding_method}")
    return embs


def init(device_que):
    os.environ["CUDA_VISIBLE_DEVICES"] = device_que.get()
    import cuml


def run_kmeans_cuml(args, X_shm_name, X_shape, X_dtype, n_clusters):
    shm = shared_memory.SharedMemory(name=X_shm_name)
    embs = np.ndarray(X_shape, dtype=X_dtype, buffer=shm.buf)

    import cuml

    kmeans = cuml.KMeans(n_clusters=n_clusters)
    kmeans.fit(embs)
    preds = kmeans.labels_
    score = -kmeans.inertia_

    return {
        "labels_pred": preds,
        "score": score,
    }


def main(args):
    docs = pd.read_csv(args.docs_path, sep="\t")
    embs = embed(docs["doc"], args)

    results = defaultdict(list)
    topics = docs["topic"].unique()
    topic2idx = pd.Series(dict(zip(topics, range(len(topics)))))
    targets = topic2idx.loc[docs["topic"]].tolist()

    n_clusters = len(topics)

    performance = defaultdict(list)

    if args.cuml:
        ngpus = torch.cuda.device_count()
        ctx = get_context("spawn")
        device_que = ctx.Queue()
        for i in range(ngpus):
            device_que.put(f"{i}")

        shm = shared_memory.SharedMemory(create=True, size=embs.nbytes)
        shared = np.ndarray(embs.shape, dtype=embs.dtype, buffer=shm.buf)
        shared[:] = embs[:]

        with ctx.Pool(ngpus, initializer=init, initargs=(device_que,)) as pool:

            handles = []
            for _ in range(args.repeats):

                process_args = (args, shm.name, embs.shape, embs.dtype, n_clusters)
                handle = pool.apply_async(run_kmeans_cuml, args=process_args)
                handles.append(handle)

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

    else:
        for _ in tqdm.trange(args.repeats):
            kmeans = KMeans(n_clusters=n_clusters, verbose=args.verbose)
            kmeans.fit(embs)
            preds = kmeans.labels_

            performance["rand_score"].append(rand_score(targets, preds))
            performance["adjusted_rand_score"].append(
                adjusted_rand_score(targets, preds)
            )
            performance["adjusted_mutual_info_score"].append(
                adjusted_mutual_info_score(targets, preds)
            )
            performance["homogeneity_score"].append(homogeneity_score(targets, preds))
            performance["normalized_mutual_info_score"].append(
                normalized_mutual_info_score(targets, preds)
            )
            performance["accuracy"].append(acc(targets, preds))
            performance["score"].append(-kmeans.inertia_)

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


def resampling(results, n=10):
    resampled = results.sample(n, replace=True)
    return resampled.sort_values("score", ascending=False).iloc[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_path", type=str)
    parser.add_argument(
        "--embedding_method",
        type=str,
        choices=["bert", "t5-encoder", "sbert"],
    )
    parser.add_argument("--embedding_model_name_or_path", type=str)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--cuml", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
