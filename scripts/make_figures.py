"""Render the generalization analyses as figures (cross-domain, temporal, scaling)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("data_analysis/analysis_results.json"))
Path("slides/figures").mkdir(parents=True, exist_ok=True)

# Four representative models, consistent colors/markers across all panels.
MODELS = ["quantile_gbm", "feature_gbdt", "gnn", "structural_prior", "glm_poisson"]
LABEL = {"quantile_gbm": "quantile_gbm", "feature_gbdt": "feature_gbdt",
         "gnn": "gnn", "structural_prior": "structural (no text)", "glm_poisson": "glm_poisson"}
COLOR = {"quantile_gbm": "#1f77b4", "feature_gbdt": "#2ca02c", "gnn": "#ff7f0e",
         "structural_prior": "#9467bd", "glm_poisson": "#8c8c8c"}
MARK = {"quantile_gbm": "o", "feature_gbdt": "s", "gnn": "^",
        "structural_prior": "D", "glm_poisson": "v"}
plt.rcParams.update({"font.size": 13})


def style(ax, title, xlabel, ylim=(0.55, 0.80)):
    ax.set_title(title, fontsize=15, fontweight="bold", color="#1F3A5F", pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("score Spearman ρ", fontsize=12)
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---- 1. Cross-domain ------------------------------------------------------
order = ["politics_2025-09", "technology_2025-09", "news_2025-09",
         "worldnews_2025-09", "Conservative_2025-09"]
xt = ["politics\n(in-dom*)", "technology", "news", "worldnews", "Conservative"]
fig, ax = plt.subplots(figsize=(6.2, 4.3))
for m in ["quantile_gbm", "feature_gbdt", "gnn", "structural_prior"]:
    ys = [d["cross_domain"][m][k]["score_rho"] for k in order]
    ax.plot(range(len(order)), ys, marker=MARK[m], color=COLOR[m], label=LABEL[m], lw=2, ms=7)
ax.axvspan(-0.4, 0.4, color="#1f77b4", alpha=0.06)
ax.set_xticks(range(len(order))); ax.set_xticklabels(xt, fontsize=10)
style(ax, "Cross-domain transfer", "test subreddit (train = politics)")
ax.legend(fontsize=9.5, loc="lower left", framealpha=0.9)
fig.text(0.5, -0.02, "*politics_2025-09 (same domain, 3 mo newer)", ha="center", fontsize=8, color="#666")
fig.tight_layout(); fig.savefig("slides/figures/cross_domain.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---- 2. Temporal ----------------------------------------------------------
torder = ["politics_2025-12_indomain", "politics_2025-09", "politics_2025-02", "politics_2024-08"]
gaps = [0, 3, 10, 16]
fig, ax = plt.subplots(figsize=(6.2, 4.3))
for m in ["quantile_gbm", "feature_gbdt", "gnn", "structural_prior"]:
    ys = [d["temporal"][m][k]["score_rho"] for k in torder]
    ax.plot(gaps, ys, marker=MARK[m], color=COLOR[m], label=LABEL[m], lw=2, ms=7)
ax.set_xticks(gaps); ax.set_xticklabels(["0", "3", "10", "16"])
ax.invert_xaxis()
style(ax, "Temporal stability", "months between train & test (politics)")
ax.legend(fontsize=9.5, loc="lower left", framealpha=0.9)
fig.tight_layout(); fig.savefig("slides/figures/temporal.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---- 3. Data-scaling ------------------------------------------------------
scales = [500, 1000, 2000, 4000, 7596]
fig, ax = plt.subplots(figsize=(6.2, 4.3))
for m in MODELS:
    ys = [d["data_scaling"][m][str(n)]["score_rho"] for n in scales]
    ax.plot(scales, ys, marker=MARK[m], color=COLOR[m], label=LABEL[m], lw=2, ms=7)
ax.set_xscale("log")
ax.set_xticks(scales); ax.set_xticklabels(["500", "1k", "2k", "4k", "7.6k"])
style(ax, "Data-scaling law", "training examples (log scale)")
ax.legend(fontsize=9.5, loc="lower right", framealpha=0.9)
fig.tight_layout(); fig.savefig("slides/figures/data_scaling.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print("wrote slides/figures/{cross_domain,temporal,data_scaling}.png")
