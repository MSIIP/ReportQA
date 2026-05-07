export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR=path/to/your/project
DATASET_PATH=path/to/ctrg_brain_zh_valid.json

MODEL_PATH=google/medgemma-1.5-4b-it
OUTPUT_PATH=${PROJECT_DIR}/work_dirs/infer/ctrg-brain-zh/medgemma-1.5-4b-it/output.json
python ${PROJECT_DIR}/src/infer/infer_medgemma_image3d.py \
    --model_id ${MODEL_PATH} \
    --input_json ${DATASET_PATH} \
    --output_json ${OUTPUT_PATH} \
    --num_slices 16
