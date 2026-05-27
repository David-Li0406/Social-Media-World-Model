"""Evaluate one or more results.jsonl files with pointwise + ranking metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..metrics.evaluate import evaluate_results, format_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+", help="one or more results.jsonl paths")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = p.parse_args(argv)

    summary: dict[str, dict] = {}
    for path in args.results:
        m = evaluate_results(path)
        name = Path(path).stem
        summary[name] = m
        if not args.json:
            print(format_report(name, m))

    if args.json:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
