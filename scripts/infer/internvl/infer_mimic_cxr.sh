export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR=path/to/your/project
DATASET_PATH=path/to/mimic_cxr_test.json
REFER_PATH=path/to/mimic_cxr_test_videos.json

MODEL_PATH=OpenGVLab/InternVL3_5-38B
WORK_DIR=${PROJECT_DIR}/work_dirs/infer/mimic-cxr/InternVL3_5-38B

swift infer \
    --model ${MODEL_PATH} \
    --val_dataset ${DATASET_PATH} \
    --result_path ${WORK_DIR}/output_swift.jsonl \
    --max_new_tokens 512 \
    --num_beams 1 \
    --temperature 0 \
    --infer_backend pt

python ${PROJECT_DIR}/src/infer/format_swift.py \
    --refer_path ${REFER_PATH} \
    --input_path ${WORK_DIR}/output_swift.jsonl \
    --output_path ${WORK_DIR}/output.json
