PROJECT_DIR=path/to/your/project
DATASET_PATH=path/to/ct_rate_valid.json
BASE_URL=xxx
API_KEY=xxx

# MODEL_NAME=gpt-4o-mini-2024-07-18
MODEL_NAME=gpt-5.4
# MODEL_NAME=claude-opus-4-6
# MODEL_NAME=gemini-3.1-pro-preview
WORK_DIR=${PROJECT_DIR}/work_dirs/infer/ct-rate/${MODEL_NAME}

python ${PROJECT_DIR}/src/infer/infer_api_image3d.py \
    --input_path ${DATASET_PATH} \
    --output_dir ${WORK_DIR}/output_api \
    --model ${MODEL_NAME} \
    --base_url ${BASE_URL} \
    --api_key ${API_KEY} \
    --num_slices 3 \
    --window_center 0 \
    --window_width 2000

python ${PROJECT_DIR}/src/infer/aggregate_api_output.py \
    --output_dir ${WORK_DIR}/output_api \
    --output_path ${WORK_DIR}/output_api.json

python ${PROJECT_DIR}/src/infer/format_api.py \
    --refer_path ${DATASET_PATH} \
    --input_path ${WORK_DIR}/output_api.json \
    --output_path ${WORK_DIR}/output.json
