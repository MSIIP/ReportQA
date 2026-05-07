PROJECT_DIR=path/to/your/project
DATASET_PATH=path/to/mimic_cxr_test.json
BASE_URL=xxx
API_KEY=xxx

# MODEL_NAME=gpt-4o-mini-2024-07-18
MODEL_NAME=gpt-5.4
# MODEL_NAME=claude-opus-4-6
# MODEL_NAME=gemini-3.1-pro-preview
WORK_DIR=${PROJECT_DIR}/work_dirs/infer/mimic-cxr/${MODEL_NAME}
REFER_PATH=path/to/mimic_cxr_test_videos.json

python ${PROJECT_DIR}/src/infer/infer_api_image.py \
    --input_path ${DATASET_PATH} \
    --output_dir ${WORK_DIR}/output_api \
    --model ${MODEL_NAME} \
    --base_url ${BASE_URL} \
    --api_key ${API_KEY}

python ${PROJECT_DIR}/src/infer/aggregate_api_output.py \
    --output_dir ${WORK_DIR}/output_api \
    --output_path ${WORK_DIR}/output_api.json

python ${PROJECT_DIR}/src/infer/format_api.py \
    --refer_path ${REFER_PATH} \
    --input_path ${WORK_DIR}/output_api.json \
    --output_path ${WORK_DIR}/output.json
