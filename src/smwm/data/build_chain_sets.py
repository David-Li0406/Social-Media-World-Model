"""Chains -> (context, stimulus, ground_truth) records.

Ported from buildChainSets.py.
"""
from __future__ import annotations

import random
from collections import deque
from pathlib import Path

from .io import read_json, strip_children, strip_solution, write_json


def find_node(node: dict, node_id: str) -> dict | None:
    queue: deque[dict] = deque([node])
    while queue:
        cur = queue.popleft()
        if cur["id"] == node_id:
            return cur
        for child in cur["children"]:
            queue.append(child)
    return None


def run(
    chains_path: str | Path,
    trees_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    chains = read_json(chains_path)
    trees = read_json(trees_path)

    objects: list[dict] = []
    for chain in chains:
        cut_point = rng.randint(0, len(chain) - 2)
        context = chain[:cut_point]
        stimulus = strip_solution(chain[cut_point])

        link_key = stimulus["link_id"].split("_", 1)[-1]
        node = None
        for comment in trees.get(link_key, []):
            if comment["id"] == chain[0]["id"]:
                node = find_node(comment, chain[cut_point]["id"])
                break
        if node is None:
            continue

        width = [strip_children(child) for child in node["children"]]
        objects.append(
            {
                "context": context,
                "stimulus": stimulus,
                "ground_truth": {
                    "score": chain[cut_point]["score"],
                    "controversiality": chain[cut_point]["controversiality"],
                    "width": width,
                },
            }
        )

    write_json(output_path, objects)
    print(f"[build_chain_sets] wrote {output_path}  records={len(objects)}")
