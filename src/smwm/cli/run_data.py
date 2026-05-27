"""Run any data-pipeline stage from a single config.

Stages:
  tree       - parquet -> trees JSON
  chains     - trees JSON -> chains JSON
  sets       - chains+trees JSON -> (context,stimulus,gt) sets
  summarize  - sets -> sim_output JSONL (collapse width list + LLM summary)
  splits     - sim_output JSONL -> train/test JSONL
  all        - run all stages in order
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import Config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument(
        "--stage",
        required=True,
        choices=["tree", "chains", "sets", "summarize", "splits", "all"],
    )
    p.add_argument(
        "--summarize-limit",
        type=int,
        default=None,
        help="Only summarize the first N records (for smoke tests).",
    )
    p.add_argument(
        "--summarize-input",
        default=None,
        help="Override input for the summarize stage (e.g. a trimmed JSON file).",
    )
    p.add_argument(
        "--splits-input",
        default=None,
        help="Override input for the splits stage (e.g. an existing sim_output).",
    )
    args = p.parse_args(argv)

    cfg = Config.load(args.config)
    paths = cfg["paths"]
    seed = int(cfg.get("seed", 42))

    if args.stage in ("tree", "all"):
        from ..data import build_tree

        build_tree.run(paths["parquet"], paths["trees"])

    if args.stage in ("chains", "all"):
        from ..data import build_chains

        build_chains.run(
            paths["trees"], paths["chains"], int(cfg.get("min_root_score", 20))
        )

    if args.stage in ("sets", "all"):
        from ..data import build_chain_sets

        build_chain_sets.run(paths["chains"], paths["trees"], paths["chain_sets"], seed=seed)

    if args.stage in ("summarize", "all"):
        from ..data import summarize_replies

        s = cfg["summarizer"]
        summarize_replies.run(
            args.summarize_input or paths["chain_sets"],
            paths["sim_output"],
            model_id=s.get("model_id", "Qwen/Qwen3-4B"),
            load_in_4bit=bool(s.get("load_in_4bit", True)),
            max_new_tokens=int(s.get("max_new_tokens", 500)),
            retries=int(s.get("retries", 3)),
            seed=seed,
            limit=args.summarize_limit,
        )

    if args.stage in ("splits", "all"):
        from ..data import build_splits

        build_splits.run(
            args.splits_input or paths["sim_output"],
            paths["train_jsonl"],
            paths["test_jsonl"],
            test_fraction=float(cfg.get("test_fraction", 0.2)),
            seed=seed,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
