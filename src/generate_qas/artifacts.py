from __future__ import annotations

import shutil
from pathlib import Path

from .models import RunArtifacts, TARGET_TYPES


def build_output_artifacts(dataset: str, languages: list[str], output_dir: Path) -> RunArtifacts:
    root = output_dir
    configs_dir = root / "configs"
    snapshots_dir = root / "snapshots"
    extract_dir = root / "01_extract_reports"
    stats_dir = root / "02_frequency_stats"
    # Knowledge tree review is a manual asset checkpoint rather than a numbered execution stage.
    knowledge_dir = root / "knowledge_tree"
    map_dir = root / "03_map_to_tree"
    qa_dir = root / "04_generate_qa"

    step1_configs = {}
    step2_configs = {}
    step1_outputs = {}
    step2_outputs = {}
    step1_stats = {}
    step3_configs = {}
    step3_outputs = {}

    for lang in languages:
        step1_configs[lang] = {}
        step2_configs[lang] = {}
        step1_outputs[lang] = {}
        step2_outputs[lang] = {}
        step1_stats[lang] = stats_dir / f"{lang}_frequency_stats.json"
        step3_configs[lang] = configs_dir / f"step3_{lang}.yaml"
        step3_outputs[lang] = qa_dir / f"qa_{lang}.jsonl"
        for task_type in TARGET_TYPES:
            step1_configs[lang][task_type] = configs_dir / f"step1_{task_type}_{lang}.yaml"
            step2_configs[lang][task_type] = configs_dir / f"step2_{task_type}_{lang}.yaml"
            step1_outputs[lang][task_type] = extract_dir / f"{task_type}_{lang}_extracted.jsonl"
            step2_outputs[lang][task_type] = {
                "raw": map_dir / f"{task_type}_{lang}_mapped_raw.jsonl",
                "cleaned": map_dir / f"{task_type}_{lang}_mapped_cleaned.jsonl",
                "dropped": map_dir / f"{task_type}_{lang}_mapped_dropped.jsonl",
            }

    return RunArtifacts(
        dataset=dataset,
        root=root,
        manifest=root / "manifest.json",
        configs_dir=configs_dir,
        snapshots_dir=snapshots_dir,
        extract_dir=extract_dir,
        stats_dir=stats_dir,
        knowledge_dir=knowledge_dir,
        map_dir=map_dir,
        qa_dir=qa_dir,
        pipeline_config_snapshot=snapshots_dir / "pipeline_config.original.yaml",
        prompt_snapshot=snapshots_dir / "prompt_bundle.yaml",
        knowledge_snapshot=knowledge_dir / "knowledge_tree.json",
        resolved_config_path=configs_dir / "pipeline_config.resolved.yaml",
        step1_configs=step1_configs,
        step2_configs=step2_configs,
        step3_configs=step3_configs,
        step1_outputs=step1_outputs,
        step1_stats=step1_stats,
        step2_outputs=step2_outputs,
        step3_outputs=step3_outputs,
    )


def ensure_run_directories(artifacts: RunArtifacts) -> None:
    for path in (
        artifacts.root,
        artifacts.configs_dir,
        artifacts.snapshots_dir,
        artifacts.extract_dir,
        artifacts.stats_dir,
        artifacts.knowledge_dir,
        artifacts.map_dir,
        artifacts.qa_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def snapshot_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
