export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR=path/to/your/project
DATASET_PATH=path/to/mimic_cxr_test.json

MODEL_ID=lingshu-medical-mllm/Lingshu-32B
OUTPUT_PATH=${PROJECT_DIR}/work_dirs/infer/mimic-cxr/Lingshu-32B/output.json
mkdir -p "$(dirname "${OUTPUT_PATH}")"
python ${PROJECT_DIR}/src/infer/infer_lingshu_image.py \
    --model-id ${MODEL_ID} \
    --data-json ${DATASET_PATH} \
    --output-json ${OUTPUT_PATH} \
    --device cuda:0 \
    --max-new-tokens 512 \
    --temperature 0
