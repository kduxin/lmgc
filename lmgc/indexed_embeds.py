from __future__ import annotations
from typing import Tuple, List, Mapping, Set, Hashable
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm.auto import tqdm
import time
import h5py
import numpy as np
import pandas as pd
from numpy import ndarray

from .fromstring import fromstring


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result

    return wrapper


@dataclass
class IndexedEmbeddings:
    ids: ndarray
    embs: ndarray

    def __post_init__(self):
        assert len(self.ids) == len(self.embs)

    @classmethod
    def from_tsv(cls, path, header: bool = True):
        ids = []
        embs = []
        print(f"Loading indexed embeddings from {path} ...")
        with open(path, "rt") as f:
            for i, line in enumerate(tqdm(f)):
                if header is True and i == 0:
                    continue
                idx, embstr = line.strip().split("\t")
                ids.append(idx)
                emb = fromstring(embstr, count=-1, sep="|")
                embs.append(emb)
        print(f"Loading complete.")
        return cls(ids=np.array(ids), embs=np.stack(embs))

    def to_tsv(self, path):
        format = "{:g}"
        with open(path, "wt") as f:
            f.write(f"idx\temb\n")
            for idx, emb in zip(tqdm(self.ids), self.embs):
                embstr = "|".join(map(format.format, emb.tolist()))
                f.write(f"{idx}\t{embstr}\n")

    @classmethod
    def from_h5(cls, path):
        print(f"Loading indexed embeddings from {path} ...")
        with h5py.File(path, "r") as f:
            ids = f["ids"][:]
            embs = f["embs"][:]
        print(f"Loading complete.")
        if ids.dtype == "O":
            ids = ids.astype("str")
        return cls(ids=ids, embs=embs)

    def to_h5(self, path):
        with h5py.File(path, "w") as f:
            f.create_dataset("ids", data=self.ids)
            f.create_dataset(
                "embs", data=self.embs.astype(np.float32), dtype=np.float32
            )

    def to_pandas(self, columns=["id", "emb"]):
        return pd.DataFrame({columns[0]: self.ids, columns[1]: list(self.embs)})

    def __getitem__(self, x):
        return self.ids[x], self.embs[x]

    def __len__(self, x):
        return len(self.ids)

