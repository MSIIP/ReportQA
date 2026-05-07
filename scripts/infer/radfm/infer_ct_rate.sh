export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR=path/to/your/project
DATASET_PATH=path/to/ct_rate_valid.json

LANG_ENCODER_PATH=path/to/MedLLaMA_13B
TOKENIZER_PATH=${LANG_ENCODER_PATH}
RADFM_CKPT_PATH=path/to/RadFM/pytorch_model.bin
OUTPUT_PATH=${PROJECT_DIR}/work_dirs/infer/ct-rate/RadFM/output.json
mkdir -p "$(dirname "${OUTPUT_PATH}")"
python ${PROJECT_DIR}/src/infer/infer_radfm_image3d.py \
    --lang_encoder_path ${LANG_ENCODER_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --radfm_ckpt_path ${RADFM_CKPT_PATH} \
    --dataset_json_path ${DATASET_PATH} \
    --output_json ${OUTPUT_PATH} \
    --max_new_tokens 512
