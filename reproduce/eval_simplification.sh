DATA_DIR=data/simplification/asset.json

python predict_score.py \
    --task simplification \
    --data_path ${DATA_DIR} \
    --max_source_length 1024 \

python correlation.py \
    --task simplification \
    --dataset asset \
