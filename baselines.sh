
for dataset in R2 ; do

    DOCS_PATH=data/${dataset}/raw/docs.tsv

    for model_name in all-MiniLM-L12-v2 sentence-transformers/bert-base-nli-mean-tokens all-mpnet-base-v2 all-distilroberta-v1; do
        echo model_name=${model_name}
        python kmeans.py \
            --docs_path $DOCS_PATH \
            --embedding_method sbert \
            --embedding_model_name_or_path ${model_name} \
            --repeats 100 \
            --n_init 10 \
            --output_path results/${dataset}.kmeans.SBERT.${model_name//\//_}.tsv ;
    done

    python kmeans.py \
        --docs_path $DOCS_PATH \
        --embedding_method bert \
        --embedding_model_name_or_path bert-base-uncased \
        --repeats 100 \
        --n_init 10 \
        --output_path results/${dataset}.kmeans.BERT.bert-base-uncased.tsv \

    python kmeans.py \
        --docs_path $DOCS_PATH \
        --embedding_method t5-encoder \
        --embedding_model_name_or_path data/pretrained/all-with_prefix-t5-base-v1 \
        --repeats 100 \
        --n_init 10 \
        --output_path results/${dataset}.kmeans.t5-encoder.all-with_prefix-t5-base-v1.tsv \
        --prefix "" \

done
