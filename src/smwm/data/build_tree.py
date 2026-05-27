"""Reddit parquet -> per-post comment forest dict.

Ported from buildTree.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import write_json


def build_comment_forest_dict(df: pd.DataFrame) -> dict[str, list[dict]]:
    nodes: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        nodes[row.id] = {
            "id": row.id,
            "parent_id": row.parent_id,
            "link_id": row.link_id,
            "subreddit": row.subreddit,
            "author": row.author,
            "body": row.body,
            "score": row.score,
            "controversiality": row.controversiality,
            "created_utc": row.created_utc,
            "children": [],
        }

    posts: dict[str, list[dict]] = {}
    for row in df.itertuples(index=False):
        node = nodes[row.id]
        parent_key = row.parent_id.split("_", 1)[-1]
        if parent_key in nodes:
            nodes[parent_key]["children"].append(node)
        else:
            link_key = row.link_id.split("_", 1)[-1]
            posts.setdefault(link_key, []).append(node)
    return posts


def run(parquet_path: str | Path, output_path: str | Path) -> None:
    df = pd.read_parquet(parquet_path)
    posts = build_comment_forest_dict(df)
    write_json(output_path, posts)
    print(f"[build_tree] wrote {output_path}  posts={len(posts)}")
