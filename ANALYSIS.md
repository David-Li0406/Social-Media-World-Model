# Social Media World Model — Benchmark & Generalization Analysis

This document reports (1) an expanded baseline benchmark on the politics test
set and (2) three generalization analyses — cross-domain, temporal, and
data-scaling — that probe whether the engagement predictor behaves like a
*world model* (transfers across domains/time, improves with data).

All metrics use the numeric targets. **ρ = Spearman rank correlation**
(scale-invariant ranking quality, the headline metric); pair_acc = pairwise
ordering accuracy; F1 = macro-F1 for the binary controversiality flag. Higher
is better. Reproduce with `scripts/run_analysis.py`; raw numbers in
`data_analysis/analysis_results.json` and `results/main_benchmark.json`.

---

## 1. Expanded baseline benchmark (politics, 7596 train / 1900 test)

18 baselines now share one `Baseline` interface. New this round: 6 non-LLM
(`structural_prior`, `glm_poisson`, `glm_tweedie`, `quantile_gbm`, `hawkes`,
`gnn`), a domain encoder config, and 3 LLM strategies (`llm_cot`,
`llm_fewshot`, `llm_reghead`). The universal parser (`_llm_parse`) now recovers
malformed-JSON LLM outputs (the Qwen3-32B zero-shot row is re-scored on the
full 1898 vs the earlier biased 801-skip subset).

| Model | score ρ | score pair_acc | width ρ | width pair_acc | contr F1 |
|---|---:|---:|---:|---:|---:|
| constant_median | nan | 0.000 | nan | 0.000 | 0.494 |
| retrieval_tfidf | +0.274 | 0.597 | +0.215 | 0.541 | 0.514 |
| retrieval_sbert | +0.238 | 0.583 | +0.201 | 0.535 | 0.514 |
| Qwen3-4B zero-shot | +0.374 | 0.573 | +0.508 | 0.665 | 0.223 |
| Qwen3-32B zero-shot | +0.482 | 0.649 | +0.477 | 0.669 | 0.311 |
| glm_tweedie | +0.615 | 0.714 | +0.570 | 0.667 | 0.424 |
| glm_poisson | +0.640 | 0.722 | +0.568 | 0.676 | 0.424 |
| hawkes | +0.641 | 0.718 | +0.578 | 0.636 | 0.398 |
| encoder_distilbert | +0.720 | 0.761 | +0.642 | 0.699 | 0.494 |
| llm_sft Qwen3-4B (LoRA, JSON gen) | +0.724 | 0.742 | +0.541 | 0.547 | 0.545 |
| llm_reghead Qwen3-4B (LoRA + heads) | +0.731 | 0.749 | +0.663 | 0.679 | 0.494* |
| gnn (reply-tree) | +0.745 | 0.762 | +0.676 | 0.668 | 0.494 |
| structural_prior (graph-only) | +0.750 | 0.762 | +0.670 | 0.702 | 0.415 |
| feature_gbdt | +0.759 | 0.770 | +0.676 | 0.711 | 0.424 |
| **quantile_gbm (LightGBM)** | **+0.760** | **0.774** | **+0.687** | 0.660 | **0.557** |

**Findings**
- **quantile_gbm is the new best overall** — best score ρ (0.760), best width ρ
  (0.687), and best controversiality F1 (0.557). LightGBM + pinball loss beats
  the sklearn GBDT.
- **Structure carries most of the signal.** `structural_prior` uses **no text
  at all** (only depth, parent/root score, time) yet reaches score ρ=0.750 —
  within 0.01 of the best text model. The `gnn` over the reply tree matches it
  (0.745). This is a central result: engagement is largely predictable from
  conversation *structure*.
- **Fine-tuning rescues the LLM but doesn't lead.** `llm_sft` (0.724) is far
  above zero-shot Qwen3-4B (0.374) and 32B (0.482), and best-but-one on F1
  (0.545), yet a sub-second `quantile_gbm` still edges it on every ranking
  metric at a fraction of the cost.
- **A regression head beats JSON generation.** `llm_reghead` (Qwen3-4B + numeric
  heads, no text decoding) lifts the SFT model from 0.724→**0.731** on score
  and **0.541→0.663** on width, with lower MAE and **0% parse failures** (vs the
  81% malformed-JSON rate the SFT generator hit). This confirms the proposed
  "predict numbers directly" recipe is the right way to make an LLM a numeric
  world model. *(F1 0.494 marked `*`: its controversiality head collapsed to
  the majority class under unweighted CE on the 2.4%-positive label; fixed with
  class-weighted CE for future runs.)*
- **Count GLMs / Hawkes** (0.61–0.64) are solid principled baselines; the
  self-exciting `hawkes` is competitive on width (cascade size), as expected.

---

## 2. Cross-domain generalization

Models are trained on **politics** and evaluated zero-shot on other subreddits
(all test sets from **RC_2025-09**, so differences across columns isolate
*domain* shift; time is held fixed). `politics_2025-09` is the near-in-domain
reference.

**Score ρ by test domain:**

| Model | politics | news | worldnews | Conservative | technology |
|---|---:|---:|---:|---:|---:|
| structural_prior | 0.755 | 0.736 | 0.688 | 0.688 | 0.739 |
| glm_poisson | 0.642 | 0.659 | 0.611 | 0.597 | 0.664 |
| hawkes | 0.647 | 0.647 | 0.601 | 0.586 | 0.653 |
| quantile_gbm | 0.745 | 0.723 | 0.685 | 0.670 | 0.737 |
| feature_gbdt | 0.750 | 0.732 | 0.688 | 0.686 | 0.745 |
| gnn | 0.757 | 0.736 | 0.695 | 0.685 | 0.752 |

**Findings**
- **Strong, graceful transfer.** The politics-trained world model loses only
  ~0.02–0.09 ρ moving to unseen subreddits. `technology` and `news` are nearly
  in-domain (ρ ≈ 0.72–0.75); `worldnews`/`Conservative` shift most (~0.68).
- **Structural and graph models transfer best** (gnn/feature_gbdt/structural
  all ≥0.685 everywhere) — structural dynamics are domain-general, whereas
  text-lexical signal would not be.
- Controversiality F1 actually *rises* on some domains (e.g. Conservative
  ≈0.50 vs politics ≈0.42) because the positive-class base rate differs — a
  reminder that F1 here is base-rate-sensitive.

---

## 3. Temporal / distribution shift

Models trained on politics (RC_2025-12) evaluated on politics from earlier
months — same domain, increasing time gap.

**Score ρ (width ρ in parentheses):**

| Model | 2025-12 (in-domain) | 2025-09 (−3mo) | 2025-02 (−10mo) | 2024-08 (−16mo) |
|---|---:|---:|---:|---:|
| structural_prior | 0.750 | 0.755 (0.654) | 0.739 (0.646) | 0.740 (0.646) |
| quantile_gbm | 0.760 | 0.745 (0.660) | 0.738 (0.657) | 0.731 (0.647) |
| feature_gbdt | 0.759 | 0.750 (0.663) | 0.741 (0.661) | 0.736 (0.640) |
| gnn | 0.746 | 0.757 (0.668) | 0.741 (0.675) | 0.740 (0.668) |
| glm_poisson | 0.640 | 0.642 (0.574) | 0.643 (0.543) | 0.625 (0.538) |
| hawkes | 0.641 | 0.647 (0.591) | 0.634 (0.552) | 0.620 (0.554) |

**Findings**
- **Engagement dynamics are temporally stable.** Going back **16 months**
  costs only ~0.02–0.03 ρ for the best models (quantile_gbm 0.760→0.731;
  feature_gbdt 0.759→0.736). The world model does **not** rot quickly — a key
  property for an offline training environment.
- Temporal drift is *smaller* than domain drift, i.e. *who* is talking (domain)
  matters more than *when* (time).

---

## 4. Data-scaling law

Models retrained on random politics subsamples (500 → 7596), evaluated on the
fixed politics test set.

**Score ρ vs. training size:**

| Model | 500 | 1000 | 2000 | 4000 | 7596 |
|---|---:|---:|---:|---:|---:|
| glm_poisson | 0.651 | 0.655 | 0.626 | 0.636 | 0.640 |
| hawkes | 0.643 | 0.636 | 0.678 | 0.678 | 0.641 |
| structural_prior | 0.708 | 0.728 | 0.737 | 0.747 | 0.750 |
| gnn | 0.710 | 0.737 | 0.739 | 0.740 | 0.742 |
| feature_gbdt | 0.722 | 0.722 | 0.747 | 0.759 | 0.759 |
| quantile_gbm | 0.698 | 0.723 | 0.731 | 0.743 | 0.760 |

**Findings**
- **Tree/structural models keep improving with data** and have **not saturated**
  at 7596 — quantile_gbm climbs monotonically 0.698→0.760, feature_gbdt
  0.722→0.759. More data should help; this motivates scaling the dataset.
- **Linear count GLMs saturate immediately** (glm_poisson ~0.64 flat from
  n=500) — they hit their capacity ceiling fast.
- **gnn is the most sample-efficient at small n** (0.710 at n=500, already
  near its ceiling), useful when target-domain data is scarce.

---

## 5. Takeaways for the world-model claim

1. **It is a world model, not a politics-memorizer**: ≤0.09 ρ drop across five
   unseen subreddits and ≤0.03 across 16 months of time shift.
2. **Conversation structure is the dominant, transferable signal** — a
   text-free structural prior (ρ=0.750) and a reply-tree GNN (0.745) rival the
   best text model, and transfer best across domains.
3. **A cheap supervised predictor (quantile_gbm / feature_gbdt) is the strongest
   and most scalable world model here**, beating both a fine-tuned Qwen3-4B and
   zero-shot Qwen3-32B on ranking — exactly the dense, fast, repeatable signal
   an RL bystander agent needs.
4. **Scaling is not saturated** — the tree models' monotonic data-scaling curve
   argues for collecting more interaction data.

### Scope notes
- Analyses use the numeric targets; `reply_summary` is omitted from the
  cross-domain/temporal sets (would need the LLM annotator, no GPU required for
  the numeric metrics).
- The heavy sweep uses the 6 fast models (instant per-record prediction);
  encoder/frozen-MLP/LLM are in the main table only (per-record neural encoding
  / GPU cost makes the 8-set × 5-scale sweep impractical on CPU).
- Cross-domain/temporal test sets are 2000 chains each, capped for balance.
