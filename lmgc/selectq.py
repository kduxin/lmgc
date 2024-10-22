import os
import argparse
import numpy as np
import pandas as pd


def main(args):
    queries = pd.read_csv(args.queries_path, sep="\t")
    np.random.seed(args.seed)
    sampled = queries.sample(args.n_samples)
    duplicated = sampled["query"].duplicated()
    if duplicated.any():
        print(f"{duplicated.sum()} duplicated queries in the sample!")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    sampled[["docid", "query"]].to_csv(
        args.output_path,
        index=False,
        sep="\t",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--n_samples", type=int)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
