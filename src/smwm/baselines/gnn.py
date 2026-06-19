"""Graph-aware baseline: message passing over the reply-tree path.

For each example the conversation context is the ancestor chain (root -> ... ->
parent) followed by the stimulus node. We treat this as a path graph, attach
per-node metadata features (score, controversiality, depth, body length,
time), run a few GraphSAGE-style message-passing layers along the path, and
predict the three numeric targets from the stimulus node's final embedding.

Implemented in plain torch (no torch_geometric): the path adjacency is a
banded matrix (each node talks to its previous and next node), so message
passing is a masked mean over neighbours, fully vectorised over the batch.
Runs on CPU.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from ._features import safe_float, safe_int
from .base import Baseline, get_context, get_ground_truth, get_stimulus
from .registry import register

NODE_FEAT_DIM = 6


def _node_feat(node: dict, is_stimulus: bool, depth: int) -> list[float]:
    body = node.get("body", "") or ""
    ts = safe_int(node.get("created_utc", 0))
    hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour if ts > 0 else 0
    # Stimulus score/controversiality are unknown at prediction time -> 0.
    score = 0.0 if is_stimulus else np.log1p(max(safe_float(node.get("score", 0.0)), 0.0))
    contr = 0.0 if is_stimulus else safe_float(node.get("controversiality", 0.0))
    return [
        score,
        contr,
        float(depth),
        np.log1p(len(body)),
        float(hour) / 23.0,
        1.0 if is_stimulus else 0.0,
    ]


def _example_nodes(record: dict, max_nodes: int) -> np.ndarray:
    ctx = get_context(record)
    stim = get_stimulus(record)
    nodes = [_node_feat(c, False, d) for d, c in enumerate(ctx)]
    nodes.append(_node_feat(stim, True, len(ctx)))
    nodes = nodes[-max_nodes:]  # keep most recent ancestors + stimulus
    arr = np.zeros((max_nodes, NODE_FEAT_DIM), dtype=np.float32)
    mask = np.zeros((max_nodes,), dtype=np.float32)
    arr[: len(nodes)] = np.asarray(nodes, dtype=np.float32)
    mask[: len(nodes)] = 1.0
    return arr, mask, len(nodes) - 1  # stimulus index = last filled


@register("gnn")
class GNNBaseline(Baseline):
    def __init__(self, max_nodes: int = 12, hidden: int = 64, layers: int = 2,
                 epochs: int = 8, batch_size: int = 64, lr: float = 1e-3,
                 device: str | None = None, **kwargs):
        self.max_nodes = int(max_nodes)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.device = device
        self._model: Any = None

    def _build(self):
        import torch
        from torch import nn

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        H, L = self.hidden, self.layers

        class PathGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(NODE_FEAT_DIM, H)
                self.msg = nn.ModuleList([nn.Linear(2 * H, H) for _ in range(L)])
                self.score_head = nn.Linear(H, 1)
                self.width_head = nn.Linear(H, 1)
                self.contr_head = nn.Linear(H, 2)

            def forward(self, x, mask):
                # x: (B, N, F), mask: (B, N)
                h = torch.relu(self.embed(x))
                m = mask.unsqueeze(-1)
                for layer in self.msg:
                    # neighbour = mean of previous + next node along the path
                    prev = torch.roll(h, shifts=1, dims=1)
                    nxt = torch.roll(h, shifts=-1, dims=1)
                    neigh = (prev + nxt) / 2.0
                    h = torch.relu(layer(torch.cat([h, neigh], dim=-1))) * m
                return h

        self._torch = torch
        self._model = PathGNN().to(self.device)

    def _batch(self, records):
        X = np.zeros((len(records), self.max_nodes, NODE_FEAT_DIM), dtype=np.float32)
        M = np.zeros((len(records), self.max_nodes), dtype=np.float32)
        idx = np.zeros((len(records),), dtype=np.int64)
        for i, r in enumerate(records):
            a, m, si = _example_nodes(r, self.max_nodes)
            X[i], M[i], idx[i] = a, m, si
        return X, M, idx

    def fit(self, train_records: list[dict]) -> None:
        self._build()
        torch = self._torch
        from torch.optim import Adam

        usable = [r for r in train_records if get_ground_truth(r)]
        ys, yw, yc = [], [], []
        for r in usable:
            g = get_ground_truth(r)
            ys.append(np.log1p(max(float(g.get("score", 0)), 0.0)))
            yw.append(np.log1p(max(float(g.get("width", 0)), 0.0)))
            yc.append(int(g.get("controversiality", 0)))
        ys = np.asarray(ys, np.float32); yw = np.asarray(yw, np.float32); yc = np.asarray(yc, np.int64)

        opt = Adam(self._model.parameters(), lr=self.lr)
        huber = torch.nn.SmoothL1Loss(); ce = torch.nn.CrossEntropyLoss()
        n = len(usable)
        self._model.train()
        for ep in range(self.epochs):
            order = np.random.permutation(n)
            total = 0.0
            for s in range(0, n, self.batch_size):
                bi = order[s : s + self.batch_size]
                recs = [usable[j] for j in bi]
                X, M, idx = self._batch(recs)
                Xt = torch.tensor(X, device=self.device)
                Mt = torch.tensor(M, device=self.device)
                It = torch.tensor(idx, device=self.device)
                h = self._model(Xt, Mt)
                hs = h[torch.arange(len(bi), device=self.device), It]  # stimulus node
                ps = self._model.score_head(hs).squeeze(-1)
                pw = self._model.width_head(hs).squeeze(-1)
                pc = self._model.contr_head(hs)
                loss = (huber(ps, torch.tensor(ys[bi], device=self.device))
                        + huber(pw, torch.tensor(yw[bi], device=self.device))
                        + ce(pc, torch.tensor(yc[bi], device=self.device)))
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.item())
            print(f"[gnn] epoch {ep+1}/{self.epochs} loss={total/max(1,n//self.batch_size):.4f}")

    def predict(self, record: dict) -> dict:
        torch = self._torch
        self._model.eval()
        X, M, idx = self._batch([record])
        with torch.no_grad():
            h = self._model(torch.tensor(X, device=self.device), torch.tensor(M, device=self.device))
            hs = h[0, idx[0]]
            s = float(np.expm1(self._model.score_head(hs).item()))
            w = float(np.expm1(self._model.width_head(hs).item()))
            c = int(self._model.contr_head(hs).argmax().item())
        return {
            "score": int(round(max(s, 0))),
            "width": int(round(max(w, 0))),
            "controversiality": c,
            "reply_summary": "",
        }
