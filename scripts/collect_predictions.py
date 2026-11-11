"""Extract judge-ready summaries from a finished sweep's artifacts.

The sweep writes full prediction records (all four tasks); the judge only needs
{record_id, predicted_summary, model}. Pulling just those keeps the files that
have to travel to the runner small -- the full records are ~10x larger and the
judge never reads the numeric fields.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="downloaded artifact dir")
    ap.add_argument("--out", default="forecast_preds")
    ap.add_argument("--prefix", default="", help="only files starting with this")
    ap.add_argument("--clear", action="store_true", help="empty --out first")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for f in out.glob("*.jsonl"):
            f.unlink()
    total = 0
    for f in sorted(Path(args.run_dir).glob("*.jsonl")):
        name = f.stem
        if args.prefix and not name.startswith(args.prefix):
            continue
        rows, empty = [], 0
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            s = (r.get("predicted") or {}).get("reply_summary")
            if s and str(s).strip():
                rows.append({"record_id": r["record_id"],
                             "predicted_summary": str(s), "model": name})
            else:
                empty += 1
        if not rows:
            print(f"  {name:28s} SKIP (0 usable summaries, {empty} empty)")
            continue
        p = out / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as w:
            for r in rows:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(rows)
        print(f"  {name:28s} {len(rows):>5} summaries ({empty} empty)  {p.stat().st_size/1024:.0f} KB")
    print(f"total {total:,} summaries across {len(list(out.glob('*.jsonl')))} files -> {out}/")


if __name__ == "__main__":
    main()
