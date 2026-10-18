"""Emit post-disjoint train/eval splits for the reply-forecasting task.

The per-cell train/test tags predate the globally post-disjoint pooling, so a
long-lived thread can appear on both sides. 13 posts did. Train records on an
eval post are dropped (never the reverse) so the evaluation set stays intact.
"""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path

KEEP = ("record_id", "split", "context", "stimulus", "ground_truth", "meta",
        "replies", "reply_summary", "summarizable", "n_live_replies")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--references", required=True)
    ap.add_argument("--eval-pool", required=True)
    ap.add_argument("--out", default="train_test")
    args = ap.parse_args()

    ref = {}
    for l in open(args.references, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); ref[r["record_id"]] = r
    ev = [json.loads(l)["record_id"] for l in open(args.eval_pool, encoding="utf-8") if l.strip()]
    ev_ids = set(ev)
    ev_posts = {ref[i]["meta"]["link_id"] for i in ev_ids if i in ref}

    train, dropped = [], 0
    for i, r in ref.items():
        if i in ev_ids:
            continue
        if r["meta"]["link_id"] in ev_posts:
            dropped += 1
            continue
        train.append(r)
    evalr = [ref[i] for i in ev if i in ref]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("forecast_train", train), ("forecast_eval", evalr)):
        p = out / f"{name}.jsonl.gz"
        with gzip.open(p, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in KEEP if k in r}, ensure_ascii=False) + "\n")
        print(f"{p}: {len(rows):,} records, {p.stat().st_size/1048576:.1f} MB gz")
    print(f"dropped {dropped} train records sharing a post with eval")
    pt = {r["meta"]["link_id"] for r in train}
    print(f"post overlap after cleaning: {len(pt & ev_posts)} (must be 0)")


if __name__ == "__main__":
    main()
