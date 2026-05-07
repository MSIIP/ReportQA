export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR=path/to/your/project
DATASET_PATH=path/to/mimic_cxr_test.json

MODEL_PATH=ZJU-AI4H/Hulu-Med-32B
OUTPUT_PATH=${PROJECT_DIR}/work_dirs/infer/mimic-cxr/Hulu-Med-32B/output.json
mkdir -p "$(dirname "${OUTPUT_PATH}")"
python ${PROJECT_DIR}/src/infer/infer_hulumed_image.py \
    --model-path ${MODEL_PATH} \
    --json-path ${DATASET_PATH} \
    --output-path ${OUTPUT_PATH} \
    --device cuda:0 \
    --max-new-tokens 512 \
    --temperature 0
