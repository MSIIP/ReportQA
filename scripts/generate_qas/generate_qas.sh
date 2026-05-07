#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR=path/to/your/project

INPUT_PATH=path/to/ctrg_brain_zh_valid.json
OUTPUT_DIR=${PROJECT_DIR}/work_dirs/data/ctrg-brain-zh

PIPELINE_CONFIG_PATH=${PROJECT_DIR}/scripts/generate_qas/config/pipeline_config.yaml
PROMPT_PATH=${PROJECT_DIR}/scripts/generate_qas/config/prompt_example_zh.yaml
KNOWLEDGE_TREE_PATH=${PROJECT_DIR}/scripts/generate_qas/config/knowledge_tree.json

export PYTHONPATH=${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}
cd ${PROJECT_DIR}

# step1: extract finding/diagnosis features.
# Optional overrides: --step1-task both, --step1-max-workers 20, --step1-save-batch-size 5, --step1-max-retries 3
python -m generate_qas.cli \
    --config ${PIPELINE_CONFIG_PATH} \
    --input-path ${INPUT_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --prompt-path ${PROMPT_PATH} \
    --knowledge-tree-path ${KNOWLEDGE_TREE_PATH} \
    --languages zh \
    --steps step1

# step2: calculate feature frequency statistics.
# Optional overrides: --step2-task both, --enable-embedding-cluster, --embedding-model BAAI/bge-m3, --embedding-device cpu
python -m generate_qas.cli \
    --config ${PIPELINE_CONFIG_PATH} \
    --input-path ${INPUT_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --prompt-path ${PROMPT_PATH} \
    --knowledge-tree-path ${KNOWLEDGE_TREE_PATH} \
    --languages zh \
    --steps step2

# step3: map extracted features to the knowledge tree.
# Optional overrides: --step3-task both, --step3-include-evidence-span-in-prompt true, --step3-max-workers 32
python -m generate_qas.cli \
    --config ${PIPELINE_CONFIG_PATH} \
    --input-path ${INPUT_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --prompt-path ${PROMPT_PATH} \
    --knowledge-tree-path ${KNOWLEDGE_TREE_PATH} \
    --languages zh \
    --steps step3

# step4: generate QA records.
# Optional overrides: --step4-task both, --step4-gen-base true, --step4-gen-hier true, --step4-gen-neg true, --step4-num-distractors 3
python -m generate_qas.cli \
    --config ${PIPELINE_CONFIG_PATH} \
    --input-path ${INPUT_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --prompt-path ${PROMPT_PATH} \
    --knowledge-tree-path ${KNOWLEDGE_TREE_PATH} \
    --languages zh \
    --steps step4
