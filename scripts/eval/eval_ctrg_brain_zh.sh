PROJECT_DIR=path/to/your/project
QAS_PATH=${PROJECT_DIR}/work_dirs/data/ctrg-brain-zh/full/qas_filtered_wrong.json
MODEL=Qwen/Qwen3.5-27B
BASE_URL=http://127.0.0.1:8030/v1
API_KEY=EMPTY
LANGUAGE=zh

WORK_DIRS=(
    ${PROJECT_DIR}/work_dirs/infer/ctrg-brain-zh/claude-opus-4-6
    ${PROJECT_DIR}/work_dirs/infer/ctrg-brain-zh/CT-CHAT-70B
    ${PROJECT_DIR}/work_dirs/infer/ctrg-brain-zh/gemini-3.1-pro-preview
    ${PROJECT_DIR}/work_dirs/infer/ctrg-brain-zh/gpt-5.4
    ${PROJECT_DIR}/work_dirs/infer/ctrg-brain-zh/Hulu-Med-32B
    ${PROJECT_DIR}/work_dirs/infer/ctrg-brain-zh/InternVL3_5-38B
    ${PROJECT_DIR}/work_dirs/infer/ctrg-brain-zh/medgemma-1.5-4b-it
    ${PROJECT_DIR}/work_dirs/infer/ctrg-brain-zh/Qwen3.5-27B
    ${PROJECT_DIR}/work_dirs/infer/ctrg-brain-zh/RadFM
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

    python ${PROJECT_DIR}/src/eval/ctrg_brain_filter_abnormal.py \
        --results_path ${WORK_DIR}/results.json \
        --results_abnormal_path ${WORK_DIR}/results_abnormal.json

    python ${PROJECT_DIR}/src/eval/scoring.py \
        --results_path ${WORK_DIR}/results_abnormal.json \
        --scores_dir ${WORK_DIR}/scores_abnormal
done
