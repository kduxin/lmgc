#!/bin/bash
set -x
set -e

DOCS_PATH="data/R2/raw/docs.tsv"
CACHE_ROOT="data/R2/cache"
MODEL_PATH="doc2query/all-with_prefix-t5-base-v1"
MODEL_ID="all-with_prefix-t5-base-v1"

DOC_MAXLEN=64
QUERY_MAXLEN=64

PREFIX="text2query"
J=1024

SEED=7

# Generate queries from documents X
python -m lmgc.genq \
    --model_family t5 \
    --model_path $MODEL_PATH \
    --docs_path $DOCS_PATH \
    --output_path ${CACHE_ROOT}/${MODEL_ID}_${PREFIX}/genq.tsv \
    --doc_max_len 512 \
    --query_max_len $QUERY_MAXLEN \
    --genq_per_doc 1 \
    --prefix "${PREFIX}:" \
    --fast 0 \


# (randomly) Select a part of queries as the base queries Y
python -m lmgc.selectq \
    --queries_path ${CACHE_ROOT}/${MODEL_ID}_${PREFIX}/genq.tsv \
    --output_path ${CACHE_ROOT}/${MODEL_ID}_${PREFIX}/baseq.uniform_${J}.tsv \
    --n_samples ${J} \
    --seed $SEED \

# Calculate the log-likelihoods of the base queries Y given the documents X
## Calculate for all prefixes of each query
python -m lmgc.loglik_prefixes \
    --model_path $MODEL_PATH \
    --docs_path $DOCS_PATH \
    --queries_path ${CACHE_ROOT}/${MODEL_ID}_${PREFIX}/baseq.uniform_${J}.tsv \
    --output_path ${CACHE_ROOT}/${MODEL_ID}_${PREFIX}/baseq.uniform_${J}.loglik_prefixes.tsv \
    --doc_max_len $DOC_MAXLEN \
    --query_max_len $QUERY_MAXLEN \
    --input_prefix "${PREFIX}:" \
    --fast 1 \


## In particular, extract the log-likelihoods for the whole query
python -m lmgc.extract_loglik \
    --loglik_prefixes_path ${CACHE_ROOT}/${MODEL_ID}_${PREFIX}/baseq.uniform_${J}.loglik_prefixes.tsv \
    --output_path ${CACHE_ROOT}/${MODEL_ID}_${PREFIX}/baseq.uniform_${J}.loglik.tsv \
    --n_queries ${J} \
    --prefix_length -1 \
