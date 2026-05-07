from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_prompt_bundle(prompt_bundle_path: Path) -> dict[str, Any]:
    if not prompt_bundle_path.exists():
        raise FileNotFoundError(f"Prompt bundle not found: {prompt_bundle_path}")
    with prompt_bundle_path.open("r", encoding="utf-8") as infile:
        bundle = yaml.safe_load(infile) or {}

    normalized_bundle: dict[str, Any] = {}
    for step_name in ("step1", "step3"):
        section = bundle.get(step_name)
        if section is None:
            raise ValueError(f"Prompt bundle {prompt_bundle_path} missing `{step_name}` section")
        normalized_bundle[step_name] = section
        for task_type in ("finding", "diagnosis"):
            if task_type not in section:
                raise ValueError(f"Prompt bundle {prompt_bundle_path} missing `{step_name}.{task_type}` section")
            if not isinstance(section[task_type], dict):
                raise ValueError(f"Prompt bundle {prompt_bundle_path} `{step_name}.{task_type}` must be a language map")
    return normalized_bundle


def validate_prompt_bundle_languages(prompt_bundle: dict[str, Any], languages: list[str], prompt_bundle_path: Path) -> None:
    for step_name in ("step1", "step3"):
        for task_type in ("finding", "diagnosis"):
            lang_map = prompt_bundle[step_name][task_type]
            for lang in languages:
                if lang not in lang_map:
                    raise ValueError(
                        f"Prompt bundle {prompt_bundle_path} missing `{step_name}.{task_type}.{lang}` required by dataset languages={languages}"
                    )
