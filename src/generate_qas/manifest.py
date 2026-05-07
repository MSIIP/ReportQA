from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import PipelineConfig, RunArtifacts


def initialize_manifest(spec: PipelineConfig, artifacts: RunArtifacts) -> dict[str, Any]:
    return {
        "dataset": spec.name,
        "description": spec.description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "compatibility": spec.compatibility,
        "status": {
            "prepared": True,
            "extract_reports": "pending",
            "frequency_stats": "pending",
            "knowledge_tree_review": "pending",
            "map_to_tree": "pending",
            "generate_qa": "pending",
        },
        "spec": spec.to_dict(),
        "artifacts": artifacts.to_dict(),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(manifest, outfile, ensure_ascii=False, indent=2)
