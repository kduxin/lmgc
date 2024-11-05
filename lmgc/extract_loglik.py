import argparse
import tqdm.auto as tqdm
from .utils import (
    loglik_of_query_prefix,
    loglik_of_whole_query,
)


def main(args):

    if args.prefix_length == -1:
        docid2logprobs = loglik_of_whole_query(
            args.loglik_prefixes_path,
            n_queries=args.n_queries,
            skip=1,
        )
    else:
        assert args.prefix_length > 0
        docid2logprobs = loglik_of_query_prefix(
            args.loglik_prefixes_path,
            n_queries=args.n_queries,
            skip=1,
            prefix_length=args.prefix_length,
        )

    print("Writing...")
    with open(args.output_path, "wt") as f:
        f.write("docid\tlogprobs\n")
        docid_logprobs = sorted(docid2logprobs.items(), key=lambda x: "{:>10}".format(x[0]))
        for docid, logprob_queries in tqdm.tqdm(docid_logprobs):
            logprob_str = "|".join([f"{logp:g}" for logp in logprob_queries.tolist()])
            f.write(f"{docid}\t{logprob_str}\n")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract log probability of the full sentence (i.e., the longest prefix) from prefixes."
    )
    parser.add_argument("--loglik_prefixes_path", type=str)
    parser.add_argument(
        "--prefix_length",
        type=int,
        default=-1,
        help="""Length of the prefix to extract log-likelihood from.
        - If the length of the longest prefix is shorter than this value, extract loglik from the longest prefix.
        - If set to -1, extract loglik from the longest prefix without verifying the length of the longest prefix.
        """,
    )
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--n_queries", type=int)
    parser.add_argument(
        "--query_max_len",
        type=int,
        default=None,
        help="Deprecated. Use --prefix_length instead.",
    )
    args = parser.parse_args()

    if args.query_max_len is not None:
        print("Warning: --query_max_len is deprecated. Use --prefix_length instead.")
        args.prefix_length = args.query_max_len

    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)
