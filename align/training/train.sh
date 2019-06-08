# Config Path
PIPELINE_CONFIG="data/ssd_mobilenet_v1_pill.config"

# Model Path
MY_MODEL_DIR="training"

mkdir ${MY_MODEL_DIR}

python model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG} \
    --model_dir=${MY_MODEL_DIR} \
    --num_train_steps=50000 \
    --num_eval_steps=2000 \
    --alsologtostderr



