"""End-to-end evaluation: results.jsonl -> dict of metric -> value."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..data.io import read_jsonl
from .pointwise import f1_binary_macro, mae, mse
from .ranking import pairwise_accuracy, spearman


def evaluate_results(path: str | Path) -> dict[str, Any]:
    records = read_jsonl(path)

    pred_score, true_score = [], []
    pred_width, true_width = [], []
    pred_contr, true_contr = [], []
    skipped = 0
    for r in records:
        p = r.get("predicted")
        g = r.get("ground_truth")
        if p is None or g is None:
            skipped += 1
            continue
        try:
            pred_score.append(float(p["score"]))
            true_score.append(float(g["score"]))
            pred_width.append(float(p["width"]))
            true_width.append(float(g["width"]))
            pred_contr.append(int(p["controversiality"]))
            true_contr.append(int(g["controversiality"]))
        except (KeyError, TypeError, ValueError):
            skipped += 1

    out: dict[str, Any] = {
        "n": len(pred_score),
        "skipped": skipped,
        "score": {
            "mse": mse(pred_score, true_score),
            "mae": mae(pred_score, true_score),
            **spearman(pred_score, true_score),
            "pairwise_accuracy": pairwise_accuracy(pred_score, true_score),
        },
        "width": {
            "mse": mse(pred_width, true_width),
            "mae": mae(pred_width, true_width),
            **spearman(pred_width, true_width),
            "pairwise_accuracy": pairwise_accuracy(pred_width, true_width),
        },
        "controversiality": {
            "f1_macro": f1_binary_macro(pred_contr, true_contr),
            "pos_rate_true": (sum(true_contr) / len(true_contr)) if true_contr else 0.0,
            "pos_rate_pred": (sum(pred_contr) / len(pred_contr)) if pred_contr else 0.0,
        },
    }
    return out


def format_report(name: str, m: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append(f"Model: {name}   N={m['n']}   skipped={m['skipped']}")
    lines.append("=" * 64)
    for tgt in ("score", "width"):
        d = m[tgt]
        lines.append(
            f"  {tgt:6s}  MSE={d['mse']:>12.3f}  MAE={d['mae']:>10.3f}  "
            f"Spearman rho={d['spearman_rho']:>+6.3f}  pair_acc={d['pairwise_accuracy']:>6.3f}"
        )
    c = m["controversiality"]
    lines.append(
        f"  controversiality  F1(macro)={c['f1_macro']:.4f}  "
        f"(true+={c['pos_rate_true']:.3f}  pred+={c['pos_rate_pred']:.3f})"
    )
    return "\n".join(lines)
