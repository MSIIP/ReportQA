PROJECT_DIR=path/to/your/project
MODEL=Qwen/Qwen3.5-27B
BASE_URL=http://127.0.0.1:8030/v1
API_KEY=EMPTY
LANGUAGE=en

WORK_DIR=${PROJECT_DIR}/work_dirs/data/radevalx/full
QAS_PATH=${WORK_DIR}/qas_filtered_wrong.json
REPORTS_PATH=${PROJECT_DIR}/data/radevalx/radevalx_pred.json

python ${PROJECT_DIR}/src/filter/infer_api_reportqa.py \
    --reports_path ${REPORTS_PATH} \
    --qas_path ${QAS_PATH} \
    --results_path ${WORK_DIR}/results_pred.json \
    --language ${LANGUAGE} \
    --model ${MODEL} \
    --base_url ${BASE_URL} \
    --api_key ${API_KEY}

python ${PROJECT_DIR}/src/eval/scoring.py \
    --results_path ${WORK_DIR}/results_pred.json \
    --scores_dir ${WORK_DIR}/scores_pred
