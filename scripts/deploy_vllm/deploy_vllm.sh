export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR=path/to/your/project
MODEL_PATH=Qwen/Qwen3.5-27B
PORT=8030
TENSOR_PARALLEL_SIZE=1  # align with the number of GPUs used (CUDA_VISIBLE_DEVICES)

vllm serve ${MODEL_PATH} \
    --host 127.0.0.1 \
    --port ${PORT} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --max-model-len 32768 \
    --language-model-only \
    --max-cudagraph-capture-size 128 \
    --chat-template ${PROJECT_DIR}/scripts/deploy_vllm/qwen3_nonthinking.jinja
#     --reasoning-parser qwen3
