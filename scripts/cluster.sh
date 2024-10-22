set -x
set -e

python -m lmgc.cluster \
    --docs_path data/R2/raw/docs.tsv \
    --embeddings_path data/R2/cache/all-with_prefix-t5-base-v1_text2query/baseq.uniform_1024.loglik.tsv \
    --clustering_method lmgc \
    --repeats 100 \
    --n_init 10 \
    --output_path results/R2.lmgc.tsv \
    --power 0.25 \
    --logp_clip 5.0 \
    --verbose 0 \
    --phi_estimate_method power2 \
    -J 1024 \