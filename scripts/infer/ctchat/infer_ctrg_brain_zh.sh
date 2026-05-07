export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR=path/to/your/project
DATASET_PATH=path/to/ctrg_brain_zh_valid.json

MODEL_PATH=path/to/CT-RATE/models/CT-CHAT/llava-lora-llama_3.1_70b
MODEL_BASE=path/to/Meta-Llama-3.1-70B-Instruct
ENCODER_CKPT=path/to/CT-CLIP_v2.pt
OUTPUT_PATH=${PROJECT_DIR}/work_dirs/infer/ctrg-brain-zh/CT-CHAT-70B/output.json
mkdir -p "$(dirname "${OUTPUT_PATH}")"
python -m ${PROJECT_DIR}/src/infer/infer_ctchat_image3d \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --encoder-ckpt ${ENCODER_CKPT} \
    --data-json ${DATASET_PATH} \
    --output-json ${OUTPUT_PATH} \
    --max-new-tokens 512 \
    --device cuda:0
