import json
import numpy as np

RESULTS_FILES = {
    "Qwen3-32B": "results/Qwen_Qwen3-32B_results.jsonl",
    "Qwen3-4B":  "results/Qwen_Qwen3-4B_results.jsonl",
}


def load_results(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def mse(preds, truths):
    arr_p = np.array(preds, dtype=float)
    arr_t = np.array(truths, dtype=float)
    return float(np.mean((arr_p - arr_t) ** 2))


def f1_binary_macro(preds, truths):
    tp = fp = fn = tn = 0
    for p, t in zip(preds, truths):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 1:
            fn += 1
        else:
            tn += 1
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_pos    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0.0

    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_neg    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0.0

    return (f1_pos + f1_neg) / 2


def evaluate(model_name, path):
    records = load_results(path)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}  ({len(records)} records)")
    print(f"{'='*60}")

    pred_scores, true_scores = [], []
    pred_widths, true_widths = [], []
    pred_controv, true_controv = [], []

    skipped = 0
    for r in records:
        p = r["predicted"]
        g = r["ground_truth"]
        if p is None:
            skipped += 1
            continue

        pred_scores.append(p["score"])
        true_scores.append(g["score"])

        pred_widths.append(p["width"])
        true_widths.append(g["width"])

        pred_controv.append(int(p["controversiality"]))
        true_controv.append(int(g["controversiality"]))

    print(f"  Valid records: {len(pred_scores)}  (skipped {skipped} with null predictions)")
    print(f"  Score MSE:               {mse(pred_scores, true_scores):.4f}")
    print(f"  Width MSE:               {mse(pred_widths, true_widths):.4f}")

    f1 = f1_binary_macro(pred_controv, true_controv)
    pos_rate_true = sum(true_controv) / len(true_controv)
    pos_rate_pred = sum(pred_controv) / len(pred_controv)
    print(f"  Controversiality F1 (macro): {f1:.4f}")
    print(f"    (ground truth positive rate: {pos_rate_true:.3f}, predicted: {pos_rate_pred:.3f})")


if __name__ == "__main__":
    for model_name, path in RESULTS_FILES.items():
        evaluate(model_name, path)
