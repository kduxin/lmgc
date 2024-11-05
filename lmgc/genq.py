import os
import shutil
import argparse
import multiprocessing
import pandas as pd
import tqdm
import torch
import transformers

torch.backends.cuda.matmul.allow_tf32 = True

cache = argparse.Namespace()


def fill_prompt_news2title(doc, eos, max_len):
    doc = " ".join(doc.split()[: max_len - 128])
    return """
News: {doc}.
Please provide a concise title for this news article.
The title should be a single sentence with no additional information.
End the title with a period. Only {eos} is allowed after the title.
Title:
""".format(
        doc=doc, eos=eos
    )


def extract_title_news2title(text):
    lines = text.split("\n")
    lines = [line.strip() for line in lines]

    i = len(lines) - 1
    while i >= 0 and not lines[i].startswith("Title:"):
        i -= 1

    if i == len(lines) - 1 or i == -1:
        print(f"Response in unexpected format: {text}")
        return None
    else:
        return lines[i + 1]


def main(args):
    docs = pd.read_csv(
        args.docs_path,
        sep="\t",
        usecols=["docid", "doc"],
        index_col="docid",
    )["doc"]
    docid2doc = dict(docs)

    ctx = multiprocessing.get_context("spawn")

    device_que = ctx.Queue()
    n_gpus = torch.cuda.device_count()
    assert n_gpus > 0, f"No GPU available."
    for i in range(n_gpus):
        device_que.put(i)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    f = open(args.output_path + ".tmp", "wt")
    f.write("docid\tquery\n")

    with ctx.Pool(n_gpus, initializer=init, initargs=(args, device_que)) as pool:
        handles = []

        samples = []
        for docid, doc in docid2doc.items():
            if args.prefix:
                doc = args.prefix + " " + doc

            samples.append((docid, doc))
            if len(samples) == args.batch_size:
                docid_batch, docs = zip(*samples)
                handle = pool.apply_async(gen_query, args=(docs,))
                handles.append((docid_batch, handle))
                samples = []
        if len(samples):
            docid_batch, docs = zip(*samples)
            handle = pool.apply_async(gen_query, args=(docs,))
            handles.append((docid_batch, handle))
            samples = []

        for docid_batch, handle in tqdm.tqdm(handles):
            queries = handle.get()
            for i, docid in enumerate(docid_batch):
                for query in queries[
                    i * args.genq_per_doc : (i + 1) * args.genq_per_doc
                ]:
                    f.write(f"{docid}\t{query}\n")

    f.close()
    shutil.move(args.output_path + ".tmp", args.output_path)


def init(args, device_que):
    print("Initializing ...")

    cache.args = args
    device = device_que.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    if args.model_family == "t5":
        cache.tokenizer = transformers.T5TokenizerFast.from_pretrained(args.model_path)
        if args.fast:
            cache.model = transformers.T5ForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
            )
        else:
            cache.model = transformers.T5ForConditionalGeneration.from_pretrained(
                args.model_path
            )
        cache.model = cache.model.to("cuda")
        cache.model.eval()
    elif args.model_family == "causallm":
        cache.tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_path, padding_side="left"
        )
        if cache.tokenizer.pad_token is None:
            cache.tokenizer.pad_token = cache.tokenizer.eos_token
        cache.model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to("cuda")
        cache.model.eval()
    else:
        raise ValueError(args.model_family)

    print("Initialization finished")


@torch.inference_mode()
def gen_query(docs):
    tokenizer = cache.tokenizer
    model = cache.model
    args = cache.args

    if args.prompt_template == "news2title":
        docs = [
            fill_prompt_news2title(doc, tokenizer.eos_token, args.doc_max_len)
            for doc in docs
        ]

    inputs = tokenizer(
        docs,
        return_tensors="pt",
        max_length=args.doc_max_len,
        truncation=True,
        padding=True,
    )
    inputs = inputs.to("cuda")

    while True:
        genq = model.generate(
            **inputs,
            do_sample=True,
            use_cache=True,
            top_k=50,
            top_p=0.99,
            max_new_tokens=args.query_max_len,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=args.genq_per_doc,
        )

        genq = tokenizer.batch_decode(sequences=genq.tolist(), skip_special_tokens=True)
        if all(len(q.strip()) > 0 for q in genq):
            break

    if args.prompt_template == "news2title":
        genq = [extract_title_news2title(q) for q in genq]
        genq = [q for q in genq if q is not None]

    return genq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_family", type=str, choices=["t5", "causallm"], default="t5"
    )
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--docs_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument(
        "--doc_max_len", type=int, default=512, help="max length of the document"
    )
    parser.add_argument(
        "--query_max_len", type=int, default=32, help="max length of the query"
    )
    parser.add_argument(
        "--genq_per_doc",
        type=int,
        default=5,
        help="Number of queries to generate per document",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument(
        "--prompt_template", type=str, default=None, choices=["news2title"]
    )
    parser.add_argument("--fast", type=int, default=0)

    args = parser.parse_args()

    print(args)

    main(args)
