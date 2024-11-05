from typing import List
import os
import argparse
import tqdm.auto as tqdm
import multiprocessing
import pandas as pd
import torch
import transformers

torch.backends.cuda.matmul.allow_tf32 = True

cache = argparse.Namespace()  # process-wise local variables


def main(args):
    docs = pd.read_csv(
        args.docs_path,
        sep="\t",
        usecols=["docid", "doc"],
    )

    queries = pd.read_csv(
        args.queries_path,
        sep="\t",
        usecols=["query"],
    )["query"].tolist()

    ctx = multiprocessing.get_context("spawn")

    device_que = ctx.Queue()
    n_gpus = torch.cuda.device_count()
    assert n_gpus > 0, f"No GPU available."
    for i in range(n_gpus):
        device_que.put(i)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    f = open(args.output_path, "wt")
    f.write("docid\tqueryid\tprefixes\n")

    with ctx.Pool(
        n_gpus, initializer=init, initargs=(args, device_que, queries)
    ) as pool:
        handles = []

        docids_batch, docs_batch = [], []
        for docid, doc in zip(docs["docid"], docs["doc"]):
            docids_batch.append(docid)
            docs_batch.append(doc)

            if len(docs_batch) == 32:
                handle = pool.apply_async(loglikelihood, args=(docs_batch,))
                handles.append((docids_batch, handle))
                docids_batch, docs_batch = [], []

        if len(docs_batch):
            handle = pool.apply_async(loglikelihood, args=(docs_batch,))
            handles.append((docids_batch, handle))
            docids_batch, docs_batch = [], []

        for docids, handle in tqdm.tqdm(handles):
            logliks_prefixes = handle.get()  # (n_docs, n_total_queries) of list
            for docid, logliks_perdoc_prefixes in zip(docids, logliks_prefixes):
                for queryid, logliks in enumerate(logliks_perdoc_prefixes):
                    f.write(
                        f"{docid}\t{queryid}\t"
                        + "|".join([f"{logl:g}" for logl in logliks])
                        + "\n"
                    )

    f.close()


def add_decoder_start_token(
    decoder_input_ids, decoder_attention_mask, decoder_start_token_id
):
    decoder_input_ids = torch.cat(
        [
            decoder_input_ids.new_full(
                (decoder_input_ids.size(0), 1), fill_value=decoder_start_token_id
            ),
            decoder_input_ids,
        ],
        dim=1,
    )
    decoder_attention_mask = torch.cat(
        [
            decoder_attention_mask.new_ones((decoder_attention_mask.size(0), 1)),
            decoder_attention_mask,
        ],
        dim=1,
    )
    return decoder_input_ids, decoder_attention_mask


def init(args, device_que, queries):
    print("Initializing ...")

    cache.args = args
    device = device_que.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    cache.tokenizer = transformers.T5TokenizerFast.from_pretrained(args.model_path)

    if args.fast:
        cache.model = transformers.T5ForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            # load_in_4bit=True,
        )
        cache.model = cache.model.to("cuda")
    else:
        cache.model = transformers.T5ForConditionalGeneration.from_pretrained(
            args.model_path
        )
        cache.model = cache.model.to("cuda")

    cache.model.eval()

    cache.queries = queries
    decoder_inputs = cache.tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cache.args.query_max_len,
    ).to("cuda")
    cache.decoder_input_ids, cache.decoder_attention_mask = add_decoder_start_token(
        decoder_inputs.input_ids,
        decoder_inputs.attention_mask,
        cache.model.config.decoder_start_token_id,
    )

    print("Initialization finished")


@torch.inference_mode()
def loglikelihood(docs: List[str]) -> List[List[float]]:

    # ------------------ preparation ------------------
    if cache.args.input_prefix:
        docs = [cache.args.input_prefix + " " + doc for doc in docs]

    inputs = cache.tokenizer(
        docs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cache.args.doc_max_len,
    ).to("cuda")
    encoder_outputs = cache.model.encoder(
        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
    )  # encoder_outputs.last_hidden_states: (docs, seqlen, hidden_size)

    n_total_queries = len(cache.queries)
    batch_size = cache.args.batch_size
    query_batch_size = batch_size // len(docs)

    # ------------------ likelihood calculation ------------------
    logliks_prefixes = [
        [None for _ in range(n_total_queries)] for _ in range(len(docs))
    ]  # (n_docs, n_total_queries) of list
    for i in range(0, n_total_queries, query_batch_size):
        n_batch_queries = len(cache.queries[i : i + query_batch_size])

        # --------- encoder (documents) ---------
        enc = encoder_outputs.copy()
        enc.last_hidden_state = (
            enc.last_hidden_state[:, None]
            .expand(-1, n_batch_queries, -1, -1)
            .flatten(0, 1)
        )  # (n_docs * n_queries, seqlen, hidden_size)

        # --------- decoder (queries) ---------
        decoder_input_ids = cache.decoder_input_ids[i : i + query_batch_size].repeat(
            len(docs), 1
        )  # (n_docs * n_queries, query_max_len)
        decoder_attention_mask = cache.decoder_attention_mask[
            i : i + query_batch_size
        ].repeat(
            len(docs), 1
        )  # (n_docs * n_queries, query_max_len)

        # --------- log p(query | document) calculation ---------
        logits = cache.model(
            encoder_outputs=enc,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
        ).logits
        logliks = (
            logits.log_softmax(dim=-1)[:, :-1, :]
            .gather(2, decoder_input_ids[..., 1:, None])
            .squeeze(-1)
        )  # (n_docs * n_queries, query_max_len)

        for j, (logl, label) in enumerate(
            zip(
                logliks.type(torch.float32).data.cpu().numpy(),
                decoder_input_ids[:, 1:].tolist(),
            )
        ):
            if 0 in label:  # truncate at the first padding token
                end = label.index(0)
            else:
                end = len(logl)

            logliks_prefixes[j // n_batch_queries][i + (j % n_batch_queries)] = logl[
                :end
            ].cumsum()
    return logliks_prefixes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        help="Huggingface model identifier or path to local checkpoints",
    )
    parser.add_argument(
        "--docs_path",
        type=str,
        help="A TSV file with `docid` and `doc` columns, where `doc` holds the text and `docid` uniquely identifies this document.",
    )
    parser.add_argument(
        "--queries_path",
        type=str,
        help="A TSV file with `query` column holding the text of every query.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Outputs a TSV. Each row contains the log-likelihood values of all prefixes of a query, conditioned on a document.",
    )
    parser.add_argument(
        "--doc_max_len",
        type=int,
        default=512,
        help="Maximum length of the document. Longer documents are truncated.",
    )
    parser.add_argument(
        "--query_max_len",
        type=int,
        default=32,
        help="Maximum length of the query. Longer queries are truncated.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="How many document-query pairs are processed within a batch",
    )
    parser.add_argument("--input_prefix", type=str, default="")
    parser.add_argument(
        "--fast",
        type=int,
        default=0,
        help="Accelerate likelihood calculation by using BetterTransformer and lower floating-point precision",
    )

    args = parser.parse_args()

    print(args)

    main(args)
