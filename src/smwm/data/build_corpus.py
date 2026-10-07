"""Multi-year corpus builder: unbiased stimulus sampling, post-disjoint splits.

Replaces the chain-based sampler for large-scale builds. Four differences from
`build_chain_sets` / `build_domain`, each fixing a measured problem in the
2025-12 dataset:

1. **width >= 0.** The chain sampler picked a cut point at
   `randint(0, len(chain)-2)`, so the stimulus was never a chain's last node and
   therefore *always* had at least one reply -- the old data has width>=1 for
   100% of records. Here stimuli are sampled from the comment tree directly and
   width is the true child count, so "nobody replied" is representable.

2. **Post-disjoint splits.** Whole `link_id`s go to train or test. In the old
   split 90% of test posts also appeared in train and 294 test records shared an
   exact stimulus id (hence label) with train.

3. **One stimulus per comment.** The old chain01 held 9,496 records over 8,433
   distinct stimuli; duplicates gave the same comment different context
   truncations, i.e. identical labels with conflicting features.

4. **Month-boundary cutoff.** A monthly dump truncates threads still receiving
   replies, so comments near the end of the file have systematically
   undercounted width. Stimuli within `tail_cutoff_days` of the file's last
   timestamp are dropped.

One streaming pass per month extracts every requested subreddit at once -- the
scan dominates the cost (~30GB/month), so it must not be repeated per subreddit.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .build_splits import make_prompt, make_target
from .build_tree import build_comment_forest_dict
from .io import strip_children, strip_solution, write_jsonl

COLS = [
    "id", "parent_id", "link_id", "subreddit", "author",
    "body", "score", "controversiality", "created_utc",
]

SECONDS_PER_DAY = 86400


# --------------------------------------------------------------- extraction
def extract_subreddits(
    parquet_path: str | Path,
    subreddits: Sequence[str],
    cache_dir: str | Path,
    month: str,
    batch_size: int = 200_000,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Stream one monthly dump once, writing a small parquet per subreddit.

    Returns {subreddit: cached_path}. Cached files make every later resampling
    essentially free, which is the point: the archive scan is the expensive step.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {s: cache_dir / f"{month}__{s}.parquet" for s in subreddits}
    if not overwrite and all(p.exists() for p in out.values()):
        print(f"[extract] {month}: cache hit for all {len(subreddits)} subreddits")
        return out

    wanted = set(subreddits)
    value_set = pa.array(sorted(wanted), type=pa.string())
    keep: dict[str, list[pd.DataFrame]] = defaultdict(list)
    pf = pq.ParquetFile(parquet_path)
    seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=COLS):
        seen += batch.num_rows
        mask = pc.is_in(batch.column("subreddit"), value_set=value_set)
        if not pc.sum(mask).as_py():
            continue
        part = batch.filter(mask).to_pandas()
        for sub, grp in part.groupby("subreddit", sort=False):
            if sub in wanted:
                keep[sub].append(grp)
    for sub, path in out.items():
        df = pd.concat(keep[sub], ignore_index=True) if keep.get(sub) else pd.DataFrame(columns=COLS)
        df.to_parquet(path, index=False)
        print(f"[extract] {month}/{sub}: {len(df):,} comments -> {path.name}")
    print(f"[extract] {month}: scanned {seen:,} rows")
    return out


# ------------------------------------------------------------ record building
def _ancestors(node_by_id: dict[str, dict], node: dict) -> list[dict]:
    """Chain from the top-level comment down to (excluding) `node`."""
    chain: list[dict] = []
    cur = node
    while True:
        parent_key = str(cur.get("parent_id", "")).split("_", 1)[-1]
        parent = node_by_id.get(parent_key)
        if parent is None:
            break
        chain.append(parent)
        cur = parent
        if len(chain) > 64:  # runaway guard against malformed parent chains
            break
    return [strip_children(n) for n in reversed(chain)]


def build_records(
    df: pd.DataFrame,
    max_posts: int | None = None,
    max_records: int | None = None,
    per_post_cap: int = 8,
    min_root_score: int = 0,
    tail_cutoff_days: float = 2.0,
    max_depth: int = 12,
    post_sampling: str = "uniform",
    seed: int = 42,
) -> list[dict]:
    """Sample stimuli from the comment tree, keeping width >= 0 outcomes.

    Posts are sampled first so memory stays bounded (a month of r/politics can
    be millions of comments); `per_post_cap` then keeps a single mega-thread
    from dominating. Both are recorded in `meta` so analyses can reweight.

    `post_sampling` controls a real bias trade-off:
      "uniform" -- every post equally likely. Maximizes thread diversity, but
        over-represents small threads relative to the comment population, which
        shifts the width distribution toward 0 (measured on 2012-10 politics:
        69% width=0 in the sample vs 56% over all comments in the month).
      "pps"     -- probability proportional to thread size, which makes the
        sample comment-representative at the cost of more within-post clustering.
    Neither is wrong; they answer different questions. Uniform is the default
    because measured within-post ICC is low (~0.02), so the clustering that
    "pps" adds costs more than the diversity it gives up.
    """
    if df.empty:
        return []
    rng = random.Random(seed)

    cutoff = None
    if tail_cutoff_days and tail_cutoff_days > 0:
        cutoff = float(df["created_utc"].max()) - tail_cutoff_days * SECONDS_PER_DAY

    if post_sampling == "pps":
        sizes = df.groupby("link_id").size()
        k = min(max_posts or len(sizes), len(sizes))
        import numpy as np

        probs = (sizes / sizes.sum()).to_numpy()
        picked = np.random.default_rng(seed).choice(
            sizes.index.to_numpy(), size=k, replace=False, p=probs
        )
        link_ids = picked.tolist()
    elif post_sampling == "uniform":
        link_ids = df["link_id"].unique().tolist()
        rng.shuffle(link_ids)
        if max_posts is not None:
            link_ids = link_ids[:max_posts]
    else:
        raise ValueError(f"post_sampling must be uniform|pps, got {post_sampling!r}")
    df = df[df["link_id"].isin(set(link_ids))]

    posts = build_comment_forest_dict(df)

    def _walk(n, depth, out):
        out.append((n, depth))
        for ch in n["children"]:
            if depth + 1 <= max_depth:
                _walk(ch, depth + 1, out)

    # id -> node map over the forest, so ancestors can be walked upward
    flat: dict[str, dict] = {}
    for _link, roots in posts.items():
        stack = list(roots)
        while stack:
            n = stack.pop()
            flat[n["id"]] = n
            stack.extend(n["children"])
    node_by_id = flat

    records: list[dict] = []
    for link_key, roots in posts.items():
        root_score = max((r.get("score") or 0) for r in roots) if roots else 0
        if root_score < min_root_score:
            continue
        cands: list[tuple[dict, int]] = []
        for r in roots:
            _walk(r, 0, cands)
        if cutoff is not None:
            cands = [(n, d) for n, d in cands if float(n.get("created_utc") or 0) <= cutoff]
        if not cands:
            continue
        rng.shuffle(cands)
        for node, depth in cands[:per_post_cap]:
            context = _ancestors(node_by_id, node)
            records.append({
                "context": context,
                "stimulus": strip_solution(node),
                "ground_truth": {
                    "score": int(node.get("score") or 0),
                    "controversiality": int(node.get("controversiality") or 0),
                    "width": len(node["children"]),
                    "reply_summary": "",
                },
                "meta": {
                    "link_id": node.get("link_id"),
                    "subreddit": node.get("subreddit"),
                    "depth": depth,
                    "root_score": int(root_score),
                    "created_utc": int(node.get("created_utc") or 0),
                },
            })
    rng.shuffle(records)
    if max_records is not None:
        records = records[:max_records]
    return records


def to_examples(records: Iterable[dict], start_id: int = 0) -> list[dict]:
    out = []
    for idx, r in enumerate(records, start=start_id):
        out.append({
            "record_id": idx,
            "context": r["context"],
            "stimulus": r["stimulus"],
            "ground_truth": r["ground_truth"],
            "meta": r["meta"],
            "prompt": make_prompt(r["context"], r["stimulus"]),
            "completion": make_target(r["ground_truth"]),
        })
    return out


# ------------------------------------------------------------------- splits
def post_disjoint_split(
    examples: Sequence[dict], test_fraction: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Assign whole posts to one side, so no thread straddles the split."""
    rng = random.Random(seed)
    by_post: dict[str, list[dict]] = defaultdict(list)
    for e in examples:
        by_post[e["meta"]["link_id"]].append(e)
    posts = sorted(by_post)
    rng.shuffle(posts)
    n_test = int(len(posts) * test_fraction)
    test_posts = set(posts[:n_test])
    train = [e for p in posts if p not in test_posts for e in by_post[p]]
    test = [e for p in sorted(test_posts) for e in by_post[p]]
    return train, test


def write_split(out_dir: str | Path, tag: str, train: list[dict], test: list[dict]) -> dict:
    out_dir = Path(out_dir)
    write_jsonl(out_dir / f"{tag}_train.jsonl", train)
    write_jsonl(out_dir / f"{tag}_test.jsonl", test)
    return {"tag": tag, "train": len(train), "test": len(test)}
