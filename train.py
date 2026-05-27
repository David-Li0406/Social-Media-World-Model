"""Top-level training entry for the GH Actions self-hosted GPU runner.

Driven by `.github/workflows/remote-gpu-exp.yml`, which invokes:

    python train.py \
        --config configs/baseline/llm_sft_qwen3_4b.yaml \
        --output_dir runs/<run_name> \
        [--train ...] [--test ...] [--data_root ...]

After training, this script:
  - calls baseline.fit() on the train JSONL (materialises the LoRA adapter
    under params.adapter_path, by default /scratch/daweili5/smwm/runs/...);
  - emits runs/<run_name>/metrics.json with status + adapter path so the
    workflow can upload it as a small artifact;
  - copies the training config alongside metrics.json for reproducibility.

Note: the LoRA adapter itself is NOT placed in runs/<run_name>/ — that
directory is for small artifacts only (per project CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Allow `import smwm.*` when the package isn't installed yet.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from smwm.baselines import registry  # noqa: E402
from smwm.config import load_yaml  # noqa: E402
from smwm.data.io import read_jsonl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", required=True, help="small-artifact output dir (runs/<run_name>)")
    p.add_argument("--train", default="train_test/RC_2025-12_politics_width_chain01_train.jsonl")
    p.add_argument("--test", default="train_test/RC_2025-12_politics_width_chain01_test.jsonl")
    p.add_argument("--data_root", default=None, help="optional override for dataset root")
    p.add_argument("--train_limit", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_yaml(args.config)
    name = cfg["baseline"]
    params = dict(cfg.get("params", {}) or {})

    train_path = Path(args.train)
    if args.data_root:
        candidate = Path(args.data_root) / Path(args.train).name
        if candidate.exists():
            train_path = candidate

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "config.yaml")

    BaselineCls = registry.get(name)
    bl = BaselineCls(**params)

    records = read_jsonl(train_path)
    if args.train_limit:
        records = records[: args.train_limit]
    print(f"[train] baseline={name}  train_records={len(records)}")

    t0 = time.time()
    metrics: dict = {
        "baseline": name,
        "train_path": str(train_path),
        "n_train": len(records),
        "status": "started",
    }
    try:
        bl.fit(records)
        metrics["status"] = "ok"
        adapter_path = getattr(bl, "adapter_path", None)
        if adapter_path is not None:
            metrics["adapter_path"] = str(adapter_path)
    except Exception as e:  # noqa: BLE001
        metrics["status"] = "error"
        metrics["error"] = repr(e)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        raise
    finally:
        metrics["elapsed_seconds"] = round(time.time() - t0, 1)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train] wrote {output_dir/'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
