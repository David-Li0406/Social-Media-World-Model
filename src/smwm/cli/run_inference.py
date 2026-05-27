"""Train+predict any registered baseline; dump results.jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..baselines import registry
from ..config import load_yaml
from ..data.io import iter_jsonl, read_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to a configs/baseline/*.yaml")
    p.add_argument("--train", required=True, help="train JSONL")
    p.add_argument("--test", required=True, help="test JSONL")
    p.add_argument(
        "--results",
        default=None,
        help="output JSONL path; defaults to results/<baseline>.jsonl",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only run on the first N test records (smoke tests).",
    )
    p.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="only fit on the first N train records (smoke tests).",
    )
    args = p.parse_args(argv)

    cfg = load_yaml(args.config)
    name = cfg["baseline"]
    params = cfg.get("params", {}) or {}

    BaselineCls = registry.get(name)
    bl = BaselineCls(**params)

    print(f"[infer] baseline={name}  params={params}")
    train_records = read_jsonl(args.train)
    if args.train_limit is not None:
        train_records = train_records[: args.train_limit]
    print(f"[infer] fitting on {len(train_records)} train records")
    bl.fit(train_records)

    out_path = Path(args.results) if args.results else Path("results") / f"{name}_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    rows = []
    for i, rec in enumerate(iter_jsonl(args.test)):
        if args.limit is not None and i >= args.limit:
            break
        pred = bl.predict(rec)
        raw = pred.pop("_raw", None)
        # ground_truth: prefer structured field, else parse completion
        gt = rec.get("ground_truth")
        if gt is None and "completion" in rec:
            try:
                gt = json.loads(rec["completion"])
            except (TypeError, ValueError):
                gt = None
        row = {
            "record_id": rec.get("record_id", i),
            "predicted": pred,
            "ground_truth": gt,
        }
        if raw is not None:
            row["raw_output"] = raw
        rows.append(row)
        n_total += 1
        if (n_total % 50) == 0:
            print(f"[infer] {n_total} done")

    write_jsonl(out_path, rows)
    print(f"[infer] wrote {out_path}  records={n_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
