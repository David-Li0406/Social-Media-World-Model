"""Phase A of the reply_summary target: recover the actual reply texts.

The multi-year corpus stores `width` as a count, so the replies themselves are
gone. They are recoverable from the per-(month, subreddit) parquet cache without
touching the 1.3TB archive: refilter to the posts we need, rebuild the comment
forest, and read each stimulus node's children.

A summary is only defined where replies exist, so this samples from width >= 1
(~30% of the corpus), stratified over subreddit x width bucket and spread across
months so no era or community dominates.

The output keeps the raw replies rather than only a generated summary. The
project's gold `reply_summary` is Qwen3-4B output -- a model artifact, not
ground truth -- so storing the real reply texts keeps the target reproducible,
lets the summarizer be swapped or rerun, and supports metrics that never needed
an LLM. Phase B (scripts/summarize_subset.py) adds the generated summary and
needs a GPU.

Usage:
    PYTHONPATH=src python3 scripts/build_summary_subset.py --target 15000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smwm.data.build_tree import build_comment_forest_dict  # noqa: E402

CORPUS = "/scratch/daweili5/smwm/data_multiyear_v2"
CACHE = "/scratch/daweili5/smwm/corpus_cache"


def width_bucket(w: int) -> str:
    if w == 1:
        return "1"
    if w <= 3:
        return "2-3"
    if w <= 9:
        return "4-9"
    return "10+"


def load_candidates(corpus: Path):
    """Every width>=1 record, tagged with its cell and split of origin."""
    cand = []
    files = [f for f in sorted(corpus.glob("*_train.jsonl")) + sorted(corpus.glob("*_test.jsonl"))
             if not f.name.startswith(("random_", "temporal_"))]
    for f in files:
        sub, month = f.name.replace("_train.jsonl", "").replace("_test.jsonl", "").rsplit("_", 1)
        split = "train" if f.name.endswith("_train.jsonl") else "test"
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["ground_truth"]["width"] < 1:
                    continue
                cand.append({
                    "record_id": r["record_id"],
                    "month": month, "subreddit": sub, "split": split,
                    "link_id": r["meta"]["link_id"],
                    "stimulus_id": r["stimulus"]["id"],
                    "width": r["ground_truth"]["width"],
                    "context": r["context"], "stimulus": r["stimulus"],
                    "ground_truth": r["ground_truth"], "meta": r["meta"],
                })
    return cand


def stratified(cand, target, seed=42):
    """Even allocation over (subreddit x width bucket), then spread over months."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for c in cand:
        strata[(c["subreddit"], width_bucket(c["width"]))].append(c)
    per = max(1, target // max(len(strata), 1))
    chosen = []
    for key, rows in sorted(strata.items()):
        by_month = defaultdict(list)
        for r in rows:
            by_month[r["month"]].append(r)
        months = sorted(by_month)
        for m in months:
            rng.shuffle(by_month[m])
        take, i = [], 0
        # round-robin across months so one month cannot dominate a stratum
        while len(take) < min(per, len(rows)):
            m = months[i % len(months)]
            if by_month[m]:
                take.append(by_month[m].pop())
            i += 1
            if i > len(rows) * 2 + len(months):
                break
        chosen.extend(take)
    rng.shuffle(chosen)
    return chosen[:target]


def attach_replies(chosen, cache: Path, max_reply_chars: int):
    """Re-derive each stimulus's direct replies from the cached parquet."""
    import pandas as pd

    by_cell = defaultdict(list)
    for c in chosen:
        by_cell[(c["month"], c["subreddit"])].append(c)

    out, missing = [], 0
    for (month, sub), rows in sorted(by_cell.items()):
        path = cache / f"{month}__{sub}.parquet"
        if not path.exists():
            missing += len(rows)
            continue
        want_posts = {r["link_id"] for r in rows}
        df = pd.read_parquet(path)
        df = df[df["link_id"].isin(want_posts)]
        if df.empty:
            missing += len(rows)
            continue
        posts = build_comment_forest_dict(df)
        flat = {}
        for _link, roots in posts.items():
            stack = list(roots)
            while stack:
                n = stack.pop()
                flat[n["id"]] = n
                stack.extend(n["children"])
        for r in rows:
            node = flat.get(r["stimulus_id"])
            if node is None:
                missing += 1
                continue
            replies = [{
                "id": ch["id"],
                "author": ch.get("author"),
                "body": (ch.get("body") or "")[:max_reply_chars],
                "score": int(ch.get("score") or 0),
                "controversiality": int(ch.get("controversiality") or 0),
                "created_utc": int(ch.get("created_utc") or 0),
            } for ch in node["children"]]
            gt = dict(r["ground_truth"])
            # 17% of all reply bodies are [removed]/[deleted]; where EVERY reply
            # is gone there is nothing to summarize, so mark it rather than
            # silently feeding placeholders to the summarizer.
            live = [x for x in replies if x["body"] not in ("[removed]", "[deleted]")]
            # width was computed on the same forest; assert rather than trust
            if len(replies) != gt["width"]:
                gt["width_recovered"] = len(replies)
            out.append({
                "record_id": r["record_id"], "split": r["split"],
                "context": r["context"], "stimulus": r["stimulus"],
                "ground_truth": gt, "meta": r["meta"],
                "replies": replies,
                "n_live_replies": len(live),
                "summarizable": bool(live),
                "reply_summary": "",   # filled by Phase B
            })
        print(f"[replies] {month}/{sub}: {len(rows)} requested, {len(df):,} cached rows")
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--target", type=int, default=15000)
    ap.add_argument("--max-reply-chars", type=int, default=1500)
    ap.add_argument("--out", default="/scratch/daweili5/smwm/data_multiyear_v2/summary_subset.jsonl")
    args = ap.parse_args()

    t0 = time.time()
    cand = load_candidates(Path(args.corpus))
    print(f"candidates with width>=1: {len(cand):,}")
    chosen = stratified(cand, args.target)
    print(f"stratified sample: {len(chosen):,}")
    rows, missing = attach_replies(chosen, Path(args.cache), args.max_reply_chars)
    print(f"\nrecovered replies for {len(rows):,} records ({missing} unrecoverable)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import numpy as np
    w = np.array([r["ground_truth"]["width"] for r in rows])
    nrep = np.array([len(r["replies"]) for r in rows])
    mism = int((w != nrep).sum())
    chars = np.array([sum(len(x["body"]) for x in r["replies"]) for r in rows])
    print(f"\nwidth vs recovered replies: {mism} mismatches")
    print(f"replies per record: mean={nrep.mean():.2f} median={np.median(nrep):.0f} max={nrep.max()}")
    print(f"reply text per record: mean={chars.mean():.0f} chars median={np.median(chars):.0f}")
    cells = len({(r['meta']['subreddit'], r['meta'].get('link_id') and
                  __import__('datetime').datetime.utcfromtimestamp(
                      r['meta']['created_utc']).strftime('%Y-%m')) for r in rows})
    nosum = sum(1 for r in rows if not r["summarizable"])
    print(f"distinct (subreddit, month) cells covered: {cells}")
    print(f"unsummarizable (every reply removed/deleted): {nosum:,} "
          f"({100*nosum/max(len(rows),1):.1f}%)")
    print(f"wrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
