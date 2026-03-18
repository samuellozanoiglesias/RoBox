#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to run this launcher") from exc


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full RoBox integration (training + analyses + dashboard + reproducibility checks)"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Base pipeline config file")
    parser.add_argument(
        "--respect-config",
        action="store_true",
        help="Do not force training.run_training=true; use value from config as-is",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    pipeline_module = importlib.import_module("robox_octagon.pipeline")
    pipeline_main = getattr(pipeline_module, "main")

    config_path = (repo_root / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if args.respect_config:
        print(f"Launching pipeline with config: {config_path}")
        pipeline_main(config_path=str(config_path))
        return

    cfg = _load_yaml(config_path)
    cfg.setdefault("pipeline", {})
    cfg["pipeline"].setdefault("training", {})
    cfg["pipeline"]["training"]["run_training"] = True

    with tempfile.NamedTemporaryFile(prefix="robox_train_", suffix=".yaml", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    _write_yaml(tmp_path, cfg)
    print(f"Launching pipeline with generated config: {tmp_path}")
    print("training.run_training is forced to true for this run")
    pipeline_main(config_path=str(tmp_path))


if __name__ == "__main__":
    main()
