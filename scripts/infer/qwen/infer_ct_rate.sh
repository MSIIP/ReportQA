export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR=path/to/your/project
DATASET_PATH=path/to/ct_rate_valid.json
PROJECT_DIR_TRL=path/to/vlm-tutorial-trl

MODEL_PATH=Qwen/Qwen3.5-27B
WORK_DIR=${PROJECT_DIR}/work_dirs/infer/ct-rate/Qwen3.5-27B

python ${PROJECT_DIR_TRL}/sft/infer.py \
    --model_name_or_path ${MODEL_PATH} \
    --dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --dataset_name ${DATASET_PATH} \
    --output_path ${WORK_DIR}/output_trl.json \
    --video_size_t 16 \
    --image_size_h 512 \
    --image_size_w 512 \
    --window_center 0 \
    --window_width 2000 \
    --max_length 32768 \
    --max_new_tokens 512 \
    --num_beams 1 \
    --temperature 0

python ${PROJECT_DIR}/src/infer/format_trl.py \
    --refer_path ${DATASET_PATH} \
    --input_path ${WORK_DIR}/output_trl.json \
    --output_path ${WORK_DIR}/output.json
