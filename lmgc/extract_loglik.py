import argparse
import tqdm.auto as tqdm
from .loglik_prefixes_utils import extract_whole_query_loglik


def main(args):

    docid2logprobs = extract_whole_query_loglik(
        args.loglik_prefixes_path, args.n_queries, skip=1, max_len=args.query_max_len
    )

    print("Writing...")
    with open(args.output_path, "wt") as f:
        f.write("docid\tlogprobs\n")
        for docid, logprob_queries in tqdm.tqdm(docid2logprobs.items()):
            logprob_str = "|".join([f"{logp:g}" for logp in logprob_queries.tolist()])
            f.write(f"{docid}\t{logprob_str}\n")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract log probability of the full sentence (i.e., the longest prefix) from prefixes."
    )
    parser.add_argument("--loglik_prefixes_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--n_queries", type=int)
    parser.add_argument("--query_max_len", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    main(args)
