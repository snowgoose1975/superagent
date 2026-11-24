"""Explicitly regenerate executor_c/capabilities.json from the runtime registry."""

from __future__ import annotations

import json
from pathlib import Path

from executor_c.model_registry import capabilities


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "executor_c" / "capabilities.json"


def main() -> int:
    content = json.dumps(capabilities(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    OUTPUT_PATH.write_text(content, encoding="utf-8", newline="\n")
    print(f"Exported model registry capabilities to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
