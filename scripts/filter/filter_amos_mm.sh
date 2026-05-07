PROJECT_DIR=path/to/your/project
WORK_DIR=${PROJECT_DIR}/work_dirs/data/amos-mm
REPORTS_PATH=path/to/amos_mm_valid.json
VERSION=full
MODEL=Qwen/Qwen3.5-27B
BASE_URL=http://127.0.0.1:8030/v1
API_KEY=EMPTY
LANGUAGE=en

python ${PROJECT_DIR}/src/filter/jsonl2swift.py \
    --qas_path_jsonl ${WORK_DIR}/${VERSION}.jsonl \
    --qas_path ${WORK_DIR}/${VERSION}/qas_raw.json \
    --language ${LANGUAGE}

# 1. 过滤无需图像即可答对的QA
python ${PROJECT_DIR}/src/filter/filter_text_infer_api.py \
    --reports_path ${REPORTS_PATH} \
    --qas_path ${WORK_DIR}/${VERSION}/qas_raw.json \
    --results_path ${WORK_DIR}/${VERSION}/results_raw.json \
    --language ${LANGUAGE} \
    --model ${MODEL} \
    --base_url ${BASE_URL} \
    --api_key ${API_KEY}

python ${PROJECT_DIR}/src/eval/scoring.py \
    --results_path ${WORK_DIR}/${VERSION}/results_raw.json \
    --scores_dir ${WORK_DIR}/${VERSION}/scores_raw

python ${PROJECT_DIR}/src/filter/filter_text.py \
    --results_path ${WORK_DIR}/${VERSION}/results_raw.json \
    --filter_text_path ${WORK_DIR}/${VERSION}/filter_text.json

python ${PROJECT_DIR}/src/filter/filter_jsonl.py \
    --filter_path ${WORK_DIR}/${VERSION}/filter_text.json \
    --input_jsonl ${WORK_DIR}/${VERSION}.jsonl \
    --output_jsonl ${WORK_DIR}/${VERSION}/qas_filtered_text.jsonl

python ${PROJECT_DIR}/src/filter/jsonl2swift.py \
    --qas_path_jsonl ${WORK_DIR}/${VERSION}/qas_filtered_text.jsonl \
    --qas_path ${WORK_DIR}/${VERSION}/qas_filtered_text.json \
    --language ${LANGUAGE}

# 2. 过滤信息不足的QA
python ${PROJECT_DIR}/src/filter/infer_api_reportqa.py \
    --reports_path ${REPORTS_PATH} \
    --qas_path ${WORK_DIR}/${VERSION}/qas_filtered_text.json \
    --results_path ${WORK_DIR}/${VERSION}/results_filtered_text.json \
    --language ${LANGUAGE} \
    --model ${MODEL} \
    --base_url ${BASE_URL} \
    --api_key ${API_KEY}

python ${PROJECT_DIR}/src/eval/scoring.py \
    --results_path ${WORK_DIR}/${VERSION}/results_filtered_text.json \
    --scores_dir ${WORK_DIR}/${VERSION}/scores_filtered_text

python ${PROJECT_DIR}/src/filter/filter_insuff.py \
    --results_path ${WORK_DIR}/${VERSION}/results_filtered_text.json \
    --filter_insuff_path ${WORK_DIR}/${VERSION}/filter_insuff.json \
    --language ${LANGUAGE}

python ${PROJECT_DIR}/src/filter/filter_jsonl.py \
    --filter_path ${WORK_DIR}/${VERSION}/filter_insuff.json \
    --input_jsonl ${WORK_DIR}/${VERSION}/qas_filtered_text.jsonl \
    --output_jsonl ${WORK_DIR}/${VERSION}/qas_filtered_insuff.jsonl

python ${PROJECT_DIR}/src/filter/jsonl2swift.py \
    --qas_path_jsonl ${WORK_DIR}/${VERSION}/qas_filtered_insuff.jsonl \
    --qas_path ${WORK_DIR}/${VERSION}/qas_filtered_insuff.json \
    --language ${LANGUAGE}

# 3. 过滤错误的QA
python ${PROJECT_DIR}/src/filter/infer_api_reportqa.py \
    --reports_path ${REPORTS_PATH} \
    --qas_path ${WORK_DIR}/${VERSION}/qas_filtered_insuff.json \
    --results_path ${WORK_DIR}/${VERSION}/results_filtered_insuff.json \
    --language ${LANGUAGE} \
    --model ${MODEL} \
    --base_url ${BASE_URL} \
    --api_key ${API_KEY}

python ${PROJECT_DIR}/src/eval/scoring.py \
    --results_path ${WORK_DIR}/${VERSION}/results_filtered_insuff.json \
    --scores_dir ${WORK_DIR}/${VERSION}/scores_filtered_insuff

python ${PROJECT_DIR}/src/filter/filter_wrong.py \
    --results_path ${WORK_DIR}/${VERSION}/results_filtered_insuff.json \
    --filter_wrong_path ${WORK_DIR}/${VERSION}/filter_wrong.json

python ${PROJECT_DIR}/src/filter/filter_jsonl.py \
    --filter_path ${WORK_DIR}/${VERSION}/filter_wrong.json \
    --input_jsonl ${WORK_DIR}/${VERSION}/qas_filtered_insuff.jsonl \
    --output_jsonl ${WORK_DIR}/${VERSION}/qas_filtered_wrong.jsonl

python ${PROJECT_DIR}/src/filter/jsonl2swift.py \
    --qas_path_jsonl ${WORK_DIR}/${VERSION}/qas_filtered_wrong.jsonl \
    --qas_path ${WORK_DIR}/${VERSION}/qas_filtered_wrong.json \
    --language ${LANGUAGE}
