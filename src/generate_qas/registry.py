from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .models import PipelineConfig


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def load_pipeline_config(
    config_path: Path,
    *,
    input_path: str | None = None,
    output_dir: str | None = None,
    prompt_path: str | None = None,
    knowledge_tree_path: str | None = None,
    languages: list[str] | None = None,
    api_base_url: str | None = None,
    model: str | None = None,
    env_key: str | None = None,
    step_overrides: dict[str, dict[str, Any]] | None = None,
) -> PipelineConfig:
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as infile:
        raw = yaml.safe_load(infile) or {}

    _validate_pipeline_config(raw, config_path)

    merged = deepcopy(raw)
    paths = merged.setdefault("paths", {})
    if input_path:
        paths["raw_input"] = input_path
    if output_dir:
        paths["output_dir"] = output_dir
    if prompt_path:
        paths["prompt_bundle"] = prompt_path
    if knowledge_tree_path:
        paths["knowledge_tree"] = knowledge_tree_path
    if languages:
        merged["languages"] = languages

    defaults = merged.setdefault("defaults", {})
    api_defaults = defaults.setdefault("api", {})
    if api_base_url:
        api_defaults["base_url"] = api_base_url
    if model:
        api_defaults["model"] = model
    if env_key:
        api_defaults["env_key"] = env_key

    for step_name, values in (step_overrides or {}).items():
        step_defaults = defaults.setdefault(step_name, {})
        for key, value in values.items():
            if value is not None:
                step_defaults[key] = value

    spec_dir = config_path.parent
    paths = merged["paths"]
    return PipelineConfig(
        name=merged["name"],
        description=merged.get("description", ""),
        languages=[str(lang).lower() for lang in merged["languages"]],
        raw_input=_resolve_path(paths["raw_input"], spec_dir),
        prompt_bundle=_resolve_path(paths["prompt_bundle"], spec_dir),
        knowledge_tree=_resolve_path(paths["knowledge_tree"], spec_dir),
        output_dir=_resolve_path(paths["output_dir"], spec_dir),
        defaults=defaults,
        strategy=merged.get("strategy", {}),
        compatibility=merged.get("compatibility", {}),
        resolved_config=_build_resolved_config(merged, spec_dir),
        source_file=config_path,
    )


def _build_resolved_config(config: dict[str, Any], spec_dir: Path) -> dict[str, Any]:
    resolved = deepcopy(config)
    paths = resolved["paths"]
    for key in ("raw_input", "prompt_bundle", "knowledge_tree", "output_dir"):
        paths[key] = str(_resolve_path(paths[key], spec_dir))
    resolved["languages"] = [str(lang).lower() for lang in resolved["languages"]]
    return resolved


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base_dir / path).resolve()


def _validate_pipeline_config(raw: dict[str, Any], config_path: Path) -> None:
    required_top_level = {"name", "languages", "paths"}
    missing = sorted(required_top_level - set(raw))
    if missing:
        raise ValueError(f"{config_path} missing required keys: {', '.join(missing)}")

    required_paths = {"raw_input", "prompt_bundle", "knowledge_tree", "output_dir"}
    missing_paths = sorted(required_paths - set(raw["paths"]))
    if missing_paths:
        raise ValueError(f"{config_path} missing required paths: {', '.join(missing_paths)}")
