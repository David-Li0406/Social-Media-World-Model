"""Numeric-only dataset builder for a single (parquet, subreddit).

Used for cross-domain and temporal-shift analyses. Reads one monthly Reddit
dump filtered to a subreddit (predicate pushdown so 300M-row files stay
tractable), runs forest -> chains -> (context, stimulus, ground_truth), and
collapses the reply list to a count. The LLM `reply_summary` step is SKIPPED
(set to "") because the analyses only score the numeric targets — no GPU
needed. Outputs prompt/completion JSONL identical in shape to the main split.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .build_chains import build_chain
from .build_chain_sets import find_node
from .build_splits import make_prompt, make_target
from .build_tree import build_comment_forest_dict
from .io import strip_children, strip_solution, write_jsonl

# Columns the pipeline needs (avoid loading the full wide schema).
_COLS = [
    "id", "parent_id", "link_id", "subreddit", "author",
    "body", "score", "controversiality", "created_utc",
]


def load_subreddit(parquet_path: str | Path, subreddit: str,
                   max_rows: int | None = None, batch_size: int = 200_000) -> pd.DataFrame:
    """Read one subreddit from a monthly dump by STREAMING row-group batches.

    The monthly dumps are ~300M rows with a large `body` text column; a single
    read_table(filters=...) materialises far too much at once (OOM). Instead we
    iterate batches, keep only matching rows, and concatenate — peak memory is
    one batch plus the (small) accumulated subreddit slice.
    """
    pf = pq.ParquetFile(parquet_path)
    keep: list[pd.DataFrame] = []
    n = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=_COLS):
        sub = batch.column("subreddit")
        import pyarrow.compute as pc

        mask = pc.equal(sub, subreddit)
        if pc.sum(mask).as_py():
            part = batch.filter(mask).to_pandas()
            keep.append(part)
            n += len(part)
            if max_rows is not None and n >= max_rows:
                break
    if not keep:
        return pd.DataFrame(columns=_COLS)
    df = pd.concat(keep, ignore_index=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows]
    return df


def build_records(df: pd.DataFrame, min_root_score: int = 20,
                  seed: int = 42) -> list[dict]:
    """Forest -> chains -> (context, stimulus, ground_truth) with numeric width."""
    rng = random.Random(seed)
    posts = build_comment_forest_dict(df)

    # chains, with the same min-root-score gate as the main pipeline
    chains: list[list[dict]] = []
    for _link, comments in posts.items():
        for comment in comments:
            chains.extend(c for c in build_chain(comment) if c[0]["score"] >= min_root_score)

    records: list[dict] = []
    for chain in chains:
        if len(chain) < 2:
            continue
        cut = rng.randint(0, len(chain) - 2)
        context = chain[:cut]
        stimulus = strip_solution(chain[cut])

        link_key = stimulus["link_id"].split("_", 1)[-1]
        node = None
        for comment in posts.get(link_key, []):
            if comment["id"] == chain[0]["id"]:
                node = find_node(comment, chain[cut]["id"])
                break
        if node is None:
            continue
        width = len(node["children"])
        records.append({
            "context": context,
            "stimulus": stimulus,
            "ground_truth": {
                "score": chain[cut]["score"],
                "controversiality": chain[cut]["controversiality"],
                "width": width,
                "reply_summary": "",  # skipped (numeric-only)
            },
        })
    return records


def to_examples(records: list[dict]) -> list[dict]:
    out = []
    for idx, r in enumerate(records):
        out.append({
            "record_id": idx,
            "context": r["context"],
            "stimulus": r["stimulus"],
            "ground_truth": r["ground_truth"],
            "prompt": make_prompt(r["context"], r["stimulus"]),
            "completion": make_target(r["ground_truth"]),
        })
    return out


def run(parquet_path: str | Path, subreddit: str, out_dir: str | Path,
        tag: str, min_root_score: int = 20, test_fraction: float = 0.2,
        max_chains: int | None = None, max_rows: int | None = None,
        seed: int = 42) -> dict:
    """Build train/test JSONL for one (parquet, subreddit) -> out_dir/<tag>_{train,test}.jsonl."""
    out_dir = Path(out_dir)
    df = load_subreddit(parquet_path, subreddit, max_rows=max_rows)
    print(f"[build_domain] {tag}: loaded {len(df)} {subreddit} comments from {Path(parquet_path).name}")
    records = build_records(df, min_root_score=min_root_score, seed=seed)
    if max_chains is not None and len(records) > max_chains:
        rng = random.Random(seed)
        records = rng.sample(records, max_chains)
    examples = to_examples(records)

    rng = random.Random(seed)
    rng.shuffle(examples)
    split = int(len(examples) * (1 - test_fraction))
    train, test = examples[:split], examples[split:]
    write_jsonl(out_dir / f"{tag}_train.jsonl", train)
    write_jsonl(out_dir / f"{tag}_test.jsonl", test)
    print(f"[build_domain] {tag}: records={len(examples)} train={len(train)} test={len(test)}")
    return {"tag": tag, "subreddit": subreddit, "n": len(examples),
            "train": len(train), "test": len(test)}
