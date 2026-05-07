from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .artifacts import build_output_artifacts, ensure_run_directories, snapshot_file
from .legacy_configs import write_legacy_configs
from .manifest import initialize_manifest, write_manifest
from .prompts import load_prompt_bundle, validate_prompt_bundle_languages
from .registry import load_pipeline_config

TASK_CHOICES = ("finding", "diagnosis", "both")
STEP_CHOICES = ("step1", "step2", "step3", "step4")


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got `{value}`.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate report QA data from a pipeline config.")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml.")
    parser.add_argument("--output-dir", help="Override paths.output_dir in pipeline_config.yaml.")
    parser.add_argument("--input-path", help="Override paths.raw_input in pipeline_config.yaml.")
    parser.add_argument("--prompt-path", help="Override paths.prompt_bundle in pipeline_config.yaml.")
    parser.add_argument("--knowledge-tree-path", help="Override paths.knowledge_tree in pipeline_config.yaml.")
    parser.add_argument("--languages", help="Comma-separated language list, for example zh,en.")
    parser.add_argument("--steps", default="step1,step2,step3,step4", help="Comma/space-separated steps to run.")
    parser.add_argument("--api-base-url", help="Override defaults.api.base_url.")
    parser.add_argument("--model", help="Override defaults.api.model.")
    parser.add_argument("--env-key", help="Override defaults.api.env_key.")
    parser.add_argument("--step1-task", choices=TASK_CHOICES, help="Override defaults.step1_extract_reports.task.")
    parser.add_argument("--step2-task", choices=TASK_CHOICES, help="Override defaults.step2_frequency_stats.task.")
    parser.add_argument("--step3-task", choices=TASK_CHOICES, help="Override defaults.step3_map_to_tree.task.")
    parser.add_argument("--step4-task", choices=TASK_CHOICES, help="Override defaults.step4_generate_qa.task.")
    parser.add_argument("--step1-max-workers", type=int, help="Override step1 max_workers.")
    parser.add_argument("--step1-save-batch-size", type=int, help="Override step1 save_batch_size.")
    parser.add_argument("--step1-max-retries", type=int, help="Override step1 max_retries.")
    parser.add_argument("--step3-max-workers", type=int, help="Override step3 max_workers.")
    parser.add_argument("--step3-save-batch-size", type=int, help="Override step3 save_batch_size.")
    parser.add_argument("--step3-max-retries", type=int, help="Override step3 max_retries.")
    parser.add_argument("--step3-include-evidence-span-in-prompt", type=_parse_bool, help="Override step3 include_evidence_span_in_prompt.")
    parser.add_argument("--step4-num-distractors", type=int, help="Override step4 num_distractors.")
    parser.add_argument("--step4-neg-max", type=int, help="Override step4 neg_max.")
    parser.add_argument("--step4-gen-base", type=_parse_bool, help="Override step4 gen_base.")
    parser.add_argument("--step4-gen-hier", type=_parse_bool, help="Override step4 gen_hier.")
    parser.add_argument("--step4-gen-neg", type=_parse_bool, help="Override step4 gen_neg.")
    parser.add_argument("--enable-embedding-cluster", action="store_true", help="Enable step2 embedding clustering.")
    parser.add_argument("--embedding-model", help="Override step2 embedding model.")
    parser.add_argument("--embedding-device", help="Override step2 embedding device.")
    parser.add_argument("--cluster-similarity-threshold", type=float, help="Override step2 clustering threshold.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare configs and print commands without executing them.")
    return parser


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    normalized = value.replace(",", " ")
    items = [item.strip().lower() for item in normalized.split() if item.strip()]
    return items or None


def _parse_steps(raw_steps: str) -> list[str]:
    aliases = {
        "extract_reports": "step1",
        "frequency_stats": "step2",
        "map_to_tree": "step3",
        "generate_qa": "step4",
        "all": "all",
    }
    parsed = []
    for item in _split_csv(raw_steps) or []:
        step = aliases.get(item, item)
        if step == "all":
            return list(STEP_CHOICES)
        if step not in STEP_CHOICES:
            raise ValueError(f"Unsupported step `{item}`. Expected one of: {', '.join(STEP_CHOICES)}.")
        if step not in parsed:
            parsed.append(step)
    if not parsed:
        raise ValueError("No steps selected.")
    return parsed


def _step_overrides_from_args(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        "step1_extract_reports": {
            "task": args.step1_task,
            "max_workers": args.step1_max_workers,
            "save_batch_size": args.step1_save_batch_size,
            "max_retries": args.step1_max_retries,
        },
        "step2_frequency_stats": {
            "task": args.step2_task,
            "enable_embedding_cluster": True if args.enable_embedding_cluster else None,
            "embedding_model": args.embedding_model,
            "embedding_device": args.embedding_device,
            "cluster_similarity_threshold": args.cluster_similarity_threshold,
        },
        "step3_map_to_tree": {
            "task": args.step3_task,
            "max_workers": args.step3_max_workers,
            "save_batch_size": args.step3_save_batch_size,
            "max_retries": args.step3_max_retries,
            "include_evidence_span_in_prompt": args.step3_include_evidence_span_in_prompt,
        },
        "step4_generate_qa": {
            "task": args.step4_task,
            "num_distractors": args.step4_num_distractors,
            "neg_max": args.step4_neg_max,
            "gen_base": args.step4_gen_base,
            "gen_hier": args.step4_gen_hier,
            "gen_neg": args.step4_gen_neg,
        },
    }


def prepare_outputs(spec):
    prompt_bundle = load_prompt_bundle(spec.prompt_bundle)
    validate_prompt_bundle_languages(prompt_bundle, spec.languages, spec.prompt_bundle)
    artifacts = build_output_artifacts(spec.name, spec.languages, spec.output_dir)
    ensure_run_directories(artifacts)
    if spec.source_file is not None:
        snapshot_file(spec.source_file, artifacts.pipeline_config_snapshot)
    snapshot_file(spec.prompt_bundle, artifacts.prompt_snapshot)
    if spec.knowledge_tree.exists():
        snapshot_file(spec.knowledge_tree, artifacts.knowledge_snapshot)
    write_resolved_config(spec, artifacts.resolved_config_path)
    write_legacy_configs(spec, prompt_bundle, artifacts)
    manifest = initialize_manifest(spec, artifacts)
    write_manifest(artifacts.manifest, manifest)
    return artifacts


def write_resolved_config(spec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        yaml.safe_dump(spec.resolved_config, outfile, allow_unicode=True, sort_keys=False)


def _resolve_task(defaults: dict[str, object], key: str = "task", legacy_keys: tuple[str, ...] = ()) -> str:
    raw_task = defaults.get(key)
    if raw_task is None:
        for legacy_key in legacy_keys:
            if legacy_key in defaults:
                raw_task = defaults[legacy_key]
                break
    task = str(raw_task if raw_task is not None else "both").lower()
    if task == "all":
        task = "both"
    if task not in TASK_CHOICES:
        raise ValueError(f"Unsupported `{key}` value: {task}. Expected one of {TASK_CHOICES}.")
    return task


def _select_task_items(task_map: dict[str, object], task: str) -> list[tuple[str, object]]:
    return list(task_map.items()) if task == "both" else [(task, task_map[task])]


def build_step_commands(spec, artifacts, step: str) -> list[list[str]]:
    base_cmd = [sys.executable, "-m"]
    commands: list[list[str]] = []

    if step == "step1":
        step1_task = _resolve_task(spec.defaults.get("step1_extract_reports", {}))
        for _, task_map in artifacts.step1_configs.items():
            for _, config_path in _select_task_items(task_map, step1_task):
                commands.append(base_cmd + ["generate_qas.pipelines.step1_extract_reports", "--config", str(config_path)])
        return commands

    if step == "step2":
        step2_defaults = spec.defaults.get("step2_frequency_stats", {})
        step2_task = _resolve_task(step2_defaults, legacy_keys=("content",))
        for lang, stats_output in artifacts.step1_stats.items():
            cmd = base_cmd + [
                "generate_qas.pipelines.step2_frequency_stats",
                "--output_file",
                str(stats_output),
                "--task",
                step2_task,
            ]
            if step2_task in {"finding", "both"}:
                cmd.extend(["--finding_file", str(artifacts.step1_outputs[lang]["finding"])])
            if step2_task in {"diagnosis", "both"}:
                cmd.extend(["--diagnosis_file", str(artifacts.step1_outputs[lang]["diagnosis"])])
            if step2_defaults.get("enable_embedding_cluster", False):
                cmd.append("--enable_embedding_cluster")
            if step2_defaults.get("embedding_model"):
                cmd.extend(["--embedding_model", str(step2_defaults["embedding_model"])])
            if "cluster_similarity_threshold" in step2_defaults:
                cmd.extend(["--cluster_similarity_threshold", str(step2_defaults["cluster_similarity_threshold"])])
            if step2_defaults.get("embedding_device"):
                cmd.extend(["--embedding_device", str(step2_defaults["embedding_device"])])
            commands.append(cmd)
        return commands

    if step == "step3":
        step3_task = _resolve_task(spec.defaults.get("step3_map_to_tree", {}))
        for _, task_map in artifacts.step2_configs.items():
            for _, config_path in _select_task_items(task_map, step3_task):
                commands.append(base_cmd + ["generate_qas.pipelines.step3_map_to_tree", "--config", str(config_path)])
        return commands

    if step == "step4":
        for _, config_path in artifacts.step3_configs.items():
            commands.append(base_cmd + ["generate_qas.pipelines.step4_generate_qa", "--config", str(config_path)])
        return commands

    raise ValueError(f"Unsupported step: {step}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    steps = _parse_steps(args.steps)
    spec = load_pipeline_config(
        Path(args.config),
        input_path=args.input_path,
        output_dir=args.output_dir,
        prompt_path=args.prompt_path,
        knowledge_tree_path=args.knowledge_tree_path,
        languages=_split_csv(args.languages),
        api_base_url=args.api_base_url,
        model=args.model,
        env_key=args.env_key,
        step_overrides=_step_overrides_from_args(args),
    )
    artifacts = prepare_outputs(spec)
    print(f"Output directory: {artifacts.root}")
    print(f"Prepared runtime configs: {artifacts.configs_dir}")

    commands: list[list[str]] = []
    for step in steps:
        commands.extend(build_step_commands(spec, artifacts, step))

    for command in commands:
        print("Running:", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
