from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import PipelineConfig, RunArtifacts


def _get_step_defaults(defaults: dict[str, Any], key: str) -> dict[str, Any]:
    return defaults.get(key, {})


def write_legacy_configs(spec: PipelineConfig, prompt_bundle: dict[str, Any], artifacts: RunArtifacts) -> None:
    for lang in spec.languages:
        for task_type in ("finding", "diagnosis"):
            _write_yaml(
                artifacts.step1_configs[lang][task_type],
                build_step1_config(spec, prompt_bundle, lang, task_type, artifacts.step1_outputs[lang][task_type]),
            )
            _write_yaml(
                artifacts.step2_configs[lang][task_type],
                build_step2_config(
                    spec,
                    prompt_bundle,
                    lang,
                    task_type,
                    artifacts.step1_outputs[lang][task_type],
                    artifacts.step2_outputs[lang][task_type]["raw"],
                    artifacts.step2_outputs[lang][task_type]["cleaned"],
                ),
            )

        _write_yaml(
            artifacts.step3_configs[lang],
            build_step3_config(
                spec,
                lang,
                artifacts.step2_outputs[lang]["finding"]["cleaned"],
                artifacts.step2_outputs[lang]["diagnosis"]["cleaned"],
                artifacts.step3_outputs[lang],
            ),
        )


def build_step1_config(spec: PipelineConfig, prompt_bundle: dict[str, Any], lang: str, task_type: str, output_path: Path) -> dict[str, Any]:
    step1_defaults = _get_step_defaults(spec.defaults, "step1_extract_reports")
    api_defaults = spec.defaults.get("api", {})
    return {
        "setting": {
            "lang": lang,
            "target_field": task_type,
            "max_workers": step1_defaults.get("max_workers", 20),
            "save_batch_size": step1_defaults.get("save_batch_size", 5),
            "max_retries": step1_defaults.get("max_retries", 3),
        },
        "api_config": {
            "base_url": api_defaults.get("base_url", ""),
            "model": api_defaults.get("model", ""),
            "env_key": api_defaults.get("env_key", "OPENAI_API_KEY"),
        },
        "paths": {
            "input": str(spec.raw_input),
            "output": str(output_path),
        },
        "prompts": prompt_bundle["step1"],
    }


def build_step2_config(
    spec: PipelineConfig,
    prompt_bundle: dict[str, Any],
    lang: str,
    task_type: str,
    input_path: Path,
    raw_output_path: Path,
    cleaned_output_path: Path,
) -> dict[str, Any]:
    step3_defaults = _get_step_defaults(spec.defaults, "step3_map_to_tree")
    api_defaults = spec.defaults.get("api", {})
    return {
        "setting": {
            "task_type": task_type,
            "lang": lang,
            "include_evidence_span_in_prompt": step3_defaults.get("include_evidence_span_in_prompt", False),
            "max_workers": step3_defaults.get("max_workers", 32),
            "save_batch_size": step3_defaults.get("save_batch_size", 5),
            "max_retries": step3_defaults.get("max_retries", 5),
        },
        "api_config": {
            "base_url": api_defaults.get("base_url", ""),
            "model": api_defaults.get("model", ""),
            "env_key": api_defaults.get("env_key", "OPENAI_API_KEY"),
        },
        "cleaning": {
            "knowledge_config": str(spec.knowledge_tree),
        },
        "paths": {
            "input": str(input_path),
            "output": str(raw_output_path),
            "cleaned_output": str(cleaned_output_path),
        },
        "prompts": prompt_bundle["step3"],
    }


def build_step3_config(spec: PipelineConfig, lang: str, finding_input: Path, diagnosis_input: Path, output_path: Path) -> dict[str, Any]:
    step4_defaults = _get_step_defaults(spec.defaults, "step4_generate_qa")
    return {
        "setting": {
            "task": step4_defaults.get("task", step4_defaults.get("qa_type", "both")).replace("all", "both"),
            "lang": lang,
            "gen_base": step4_defaults.get("gen_base", True),
            "gen_hier": step4_defaults.get("gen_hier", True),
            "gen_neg": step4_defaults.get("gen_neg", True),
            "num_distractors": step4_defaults.get("num_distractors", 3),
            "neg_max": step4_defaults.get("neg_max", 0),
        },
        "paths": {
            "finding_input": str(finding_input),
            "diagnosis_input": str(diagnosis_input),
            "output": str(output_path),
            "knowledge_tree": str(spec.knowledge_tree),
        },
        "raw_json_files": [str(spec.raw_input)],
        "negative_fallback_leaf_nodes": spec.strategy.get("negative_fallback_leaf_nodes", {}),
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        yaml.safe_dump(payload, outfile, allow_unicode=True, sort_keys=False)
