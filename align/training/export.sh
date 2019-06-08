# Checkpoint Path
CHECKPOINT_NUMBER=$1
MY_MODEL_DIR="training"
CKPT_PREFIX=${MY_MODEL_DIR}/model.ckpt-${CHECKPOINT_NUMBER}

python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG} \
    --training_checkpoint_prefix ${CKPT_PREFIX} \
    --output_directory my_exported_graphs