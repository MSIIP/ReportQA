from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TARGET_TYPES = ("finding", "diagnosis")


@dataclass
class PipelineConfig:
    name: str
    description: str
    languages: list[str]
    raw_input: Path
    prompt_bundle: Path
    knowledge_tree: Path
    output_dir: Path
    defaults: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    compatibility: dict[str, Any] = field(default_factory=dict)
    resolved_config: dict[str, Any] = field(default_factory=dict)
    source_file: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("raw_input", "prompt_bundle", "knowledge_tree", "output_dir", "source_file"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        return data


@dataclass
class RunArtifacts:
    dataset: str
    root: Path
    manifest: Path
    configs_dir: Path
    snapshots_dir: Path
    extract_dir: Path
    stats_dir: Path
    knowledge_dir: Path
    map_dir: Path
    qa_dir: Path
    pipeline_config_snapshot: Path
    prompt_snapshot: Path
    knowledge_snapshot: Path
    resolved_config_path: Path
    step1_configs: dict[str, dict[str, Path]]
    step2_configs: dict[str, dict[str, Path]]
    step3_configs: dict[str, Path]
    step1_outputs: dict[str, dict[str, Path]]
    step1_stats: dict[str, Path]
    step2_outputs: dict[str, dict[str, dict[str, Path]]]
    step3_outputs: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "root": str(self.root),
            "manifest": str(self.manifest),
            "configs_dir": str(self.configs_dir),
            "snapshots_dir": str(self.snapshots_dir),
            "extract_dir": str(self.extract_dir),
            "stats_dir": str(self.stats_dir),
            "knowledge_dir": str(self.knowledge_dir),
            "map_dir": str(self.map_dir),
            "qa_dir": str(self.qa_dir),
            "pipeline_config_snapshot": str(self.pipeline_config_snapshot),
            "prompt_snapshot": str(self.prompt_snapshot),
            "knowledge_snapshot": str(self.knowledge_snapshot),
            "resolved_config_path": str(self.resolved_config_path),
            "step1_configs": stringify_paths(self.step1_configs),
            "step2_configs": stringify_paths(self.step2_configs),
            "step3_configs": stringify_paths(self.step3_configs),
            "step1_outputs": stringify_paths(self.step1_outputs),
            "step1_stats": stringify_paths(self.step1_stats),
            "step2_outputs": stringify_paths(self.step2_outputs),
            "step3_outputs": stringify_paths(self.step3_outputs),
        }


def stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: stringify_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [stringify_paths(v) for v in value]
    return value
