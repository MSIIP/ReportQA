PROJECT_DIR=path/to/your/project
QAS_PATH=${PROJECT_DIR}/work_dirs/data/mimic-cxr/full/qas_filtered_wrong.json
MODEL=Qwen/Qwen3.5-27B
BASE_URL=http://127.0.0.1:8030/v1
API_KEY=EMPTY
LANGUAGE=en

WORK_DIRS=(
    ${PROJECT_DIR}/work_dirs/infer/mimic-cxr/claude-opus-4-6
    ${PROJECT_DIR}/work_dirs/infer/mimic-cxr/gemini-3.1-pro-preview
    ${PROJECT_DIR}/work_dirs/infer/mimic-cxr/gpt-5.4
    ${PROJECT_DIR}/work_dirs/infer/mimic-cxr/Hulu-Med-32B
    ${PROJECT_DIR}/work_dirs/infer/mimic-cxr/InternVL3_5-38B
    ${PROJECT_DIR}/work_dirs/infer/mimic-cxr/Lingshu-32B
    ${PROJECT_DIR}/work_dirs/infer/mimic-cxr/medgemma-1.5-4b-it
    ${PROJECT_DIR}/work_dirs/infer/mimic-cxr/Qwen3.5-27B
    ${PROJECT_DIR}/work_dirs/infer/mimic-cxr/RadFM
)

for WORK_DIR in "${WORK_DIRS[@]}"; do
    python ${PROJECT_DIR}/src/filter/infer_api_reportqa.py \
        --reports_path ${WORK_DIR}/output.json \
        --qas_path ${QAS_PATH} \
        --results_path ${WORK_DIR}/results.json \
        --language ${LANGUAGE} \
        --model ${MODEL} \
        --base_url ${BASE_URL} \
        --api_key ${API_KEY}

    python ${PROJECT_DIR}/src/eval/scoring.py \
        --results_path ${WORK_DIR}/results.json \
        --scores_dir ${WORK_DIR}/scores
done
