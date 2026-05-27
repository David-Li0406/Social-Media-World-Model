"""Forest -> deduplicated reply chains.

Ported from buildChain.py (originally named buildChain.py per submit script).
"""
from __future__ import annotations

from pathlib import Path

from .io import read_json, strip_children, write_json


def build_chain(node: dict) -> list[list[dict]]:
    stack: list[tuple[dict, list[dict]]] = [(node, [strip_children(node)])]
    chains: list[list[dict]] = []

    while stack:
        cur, context = stack.pop()
        if not cur["children"]:
            chains.append(context)
        for child in cur["children"]:
            stack.append((child, context + [strip_children(child)]))

    chains.sort(key=len, reverse=True)

    used_edges: set[tuple[str, str]] = set()
    selected: list[list[dict]] = []
    for path in chains:
        edges = [(path[i]["id"], path[i + 1]["id"]) for i in range(len(path) - 1)]
        if any(e in used_edges for e in edges):
            continue
        if len(path) < 3:
            continue
        used_edges.update(edges)
        selected.append(path)
    return selected


def run(trees_path: str | Path, output_path: str | Path, min_root_score: int = 20) -> None:
    posts = read_json(trees_path)
    dump: list[list[dict]] = []
    for _link_id, comments in posts.items():
        for comment in comments:
            dump.extend(c for c in build_chain(comment) if c[0]["score"] >= min_root_score)
    write_json(output_path, dump)
    print(f"[build_chains] wrote {output_path}  chains={len(dump)}")
