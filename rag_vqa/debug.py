from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings


def debug_dump(settings: Settings, stage: str, payload: Any) -> None:
    """Print structured debug information to stderr when debug mode is enabled."""

    if not settings.debug:
        return
    print(f"\n[RAG-VQA DEBUG] {stage}", file=sys.stderr)
    print(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), file=sys.stderr)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "preview": value.flatten()[:8].round(4).tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)
