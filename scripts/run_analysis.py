"""Cross-domain, temporal, and data-scaling analyses for the world model.

Fits each baseline ONCE on the politics train split, then evaluates the same
fitted model across every shifted test set (cross-domain + temporal share the
fit). Data-scaling refits on subsamples. All metrics use the numeric targets
(score, width, controversiality) with pointwise + ranking measures.

Outputs:
  data_analysis/analysis_results.json   (machine-readable)
  printed tables for each analysis
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smwm.baselines import registry  # noqa: E402
from smwm.config import load_yaml  # noqa: E402
from smwm.data.io import read_jsonl  # noqa: E402
from smwm.metrics.pointwise import f1_binary_macro, mae, mse  # noqa: E402
from smwm.metrics.ranking import pairwise_accuracy, spearman  # noqa: E402

TRAIN = "train_test/RC_2025-12_politics_width_chain01_train.jsonl"
POLITICS_TEST = "train_test/RC_2025-12_politics_width_chain01_test.jsonl"
DOM = "data_analysis/domains"

# Fast models for the full sweep (refit cheaply). Heavier ones (frozen_mlp)
# are fit once for cross-domain/temporal but skipped in data-scaling.
FAST = ["structural_prior", "hawkes", "glm_poisson", "quantile_gbm", "feature_gbdt", "gnn"]
FITONCE = FAST + ["frozen_mlp"]

CROSS_DOMAIN = [
    ("politics_2025-09", f"{DOM}/politics_2025-09_test.jsonl"),
    ("news_2025-09", f"{DOM}/news_2025-09_test.jsonl"),
    ("worldnews_2025-09", f"{DOM}/worldnews_2025-09_test.jsonl"),
    ("Conservative_2025-09", f"{DOM}/Conservative_2025-09_test.jsonl"),
    ("technology_2025-09", f"{DOM}/technology_2025-09_test.jsonl"),
]
TEMPORAL = [
    ("politics_2025-09", f"{DOM}/politics_2025-09_test.jsonl"),
    ("politics_2025-02", f"{DOM}/politics_2025-02_test.jsonl"),
    ("politics_2024-08", f"{DOM}/politics_2024-08_test.jsonl"),
]
SCALES = [500, 1000, 2000, 4000, 7596]


def make_baseline(name):
    cfg = load_yaml(f"configs/baseline/{name}.yaml")
    return registry.get(cfg["baseline"])(**(cfg.get("params") or {}))


def evaluate(model, records):
    ps, ts, pw, tw, pc, tc = [], [], [], [], [], []
    for r in records:
        g = r.get("ground_truth") or json.loads(r["completion"])
        pred = model.predict(r)
        ps.append(float(pred["score"])); ts.append(float(g["score"]))
        pw.append(float(pred["width"])); tw.append(float(g["width"]))
        pc.append(int(pred["controversiality"])); tc.append(int(g["controversiality"]))
    return {
        "n": len(records),
        "score_rho": spearman(ps, ts)["spearman_rho"],
        "score_pacc": pairwise_accuracy(ps, ts),
        "score_mae": mae(ps, ts),
        "width_rho": spearman(pw, tw)["spearman_rho"],
        "width_pacc": pairwise_accuracy(pw, tw),
        "width_mae": mae(pw, tw),
        "contr_f1": f1_binary_macro(pc, tc),
    }


def main():
    train = read_jsonl(TRAIN)
    pol_test = read_jsonl(POLITICS_TEST)
    out = {"cross_domain": {}, "temporal": {}, "data_scaling": {}}

    # ---- fit each model once on politics train ----
    fitted = {}
    for name in FITONCE:
        print(f"[fit] {name} on {len(train)} politics records", flush=True)
        m = make_baseline(name)
        m.fit(train)
        fitted[name] = m

    # ---- cross-domain (incl. in-domain politics 2025-12 reference) ----
    domain_sets = [("politics_2025-12_indomain", POLITICS_TEST)] + CROSS_DOMAIN
    for name, m in fitted.items():
        out["cross_domain"][name] = {}
        for tag, path in domain_sets:
            recs = read_jsonl(path)
            out["cross_domain"][name][tag] = evaluate(m, recs)
            r = out["cross_domain"][name][tag]
            print(f"[cross] {name:16s} {tag:26s} score_rho={r['score_rho']:+.3f} "
                  f"width_rho={r['width_rho']:+.3f} contr_f1={r['contr_f1']:.3f}", flush=True)

    # ---- temporal (politics across months) ----
    for name, m in fitted.items():
        out["temporal"][name] = {"politics_2025-12_indomain": out["cross_domain"][name]["politics_2025-12_indomain"]}
        for tag, path in TEMPORAL:
            recs = read_jsonl(path)
            out["temporal"][name][tag] = evaluate(m, recs)
            r = out["temporal"][name][tag]
            print(f"[temporal] {name:16s} {tag:20s} score_rho={r['score_rho']:+.3f} "
                  f"width_rho={r['width_rho']:+.3f}", flush=True)

    # ---- data-scaling (fast models, refit on subsamples, eval on politics test) ----
    rng = random.Random(42)
    for name in FAST:
        out["data_scaling"][name] = {}
        for n in SCALES:
            sub = train if n >= len(train) else rng.sample(train, n)
            m = make_baseline(name)
            m.fit(sub)
            out["data_scaling"][name][str(n)] = evaluate(m, pol_test)
            r = out["data_scaling"][name][str(n)]
            print(f"[scale] {name:16s} n={n:5d} score_rho={r['score_rho']:+.3f} "
                  f"width_rho={r['width_rho']:+.3f}", flush=True)

    Path("data_analysis").mkdir(exist_ok=True)
    with open("data_analysis/analysis_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE data_analysis/analysis_results.json", flush=True)


if __name__ == "__main__":
    main()
