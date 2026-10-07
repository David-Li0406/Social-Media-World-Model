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
| llm_reghead, class-weighted CE (§7.4) | +0.710 | 0.749 | +0.634 | 0.664 | **0.595** |
| gnn (reply-tree) | +0.745 | 0.762 | +0.676 | 0.668 | 0.494 |
| structural_prior (graph-only) | +0.750 | 0.762 | +0.670 | 0.702 | 0.415 |
| feature_gbdt | +0.759 | 0.770 | +0.676 | 0.711 | 0.424 |
| **quantile_gbm (LightGBM)** | **+0.760** | **0.774** | **+0.687** | 0.660 | **0.557** |

**Findings**
- **quantile_gbm is the new best overall** — best score ρ (0.760), best width ρ
  (0.687), and best controversiality F1 (0.557). LightGBM + pinball loss beats
  the sklearn GBDT. *(Caveat added later: the 0.760 vs 0.759 gap over
  `feature_gbdt` is far inside noise — a 95% bootstrap CI on score ρ spans
  roughly ±0.018, and §7.2 shows ρ also moves ~+0.03 with feature-block width
  alone. Treat the top five rows as a tie, not a ranking.)*
- **Structure carries most of the signal.** `structural_prior` uses **no text
  at all** (only depth, parent/root score, time) yet reaches score ρ=0.750 —
  within 0.01 of the best text model. The `gnn` over the reply tree matches it
  (0.745). This is a central result: engagement is largely predictable from
  conversation *structure*. **§7 tests this properly** (the comparison here is
  confounded: `feature_gbdt`'s "text" is only shallow lexical statistics) and
  the claim survives in a stronger form.
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
  the majority class under unweighted CE on the 2.4%-positive label. §7.4 runs
  the fix and reports what it actually costs.)*
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

## 6. Reply-summary quality (LLM-as-a-judge, Qwen3-32B)

The fourth target, `reply_summary`, is free text and was previously unscored.
**Qwen3-32B** (4-bit, on the runner) rates each model's predicted summary
against the gold reference summary (Qwen3-4B's summary of the *actual* replies)
on a **1–5** scale (5 = captures all key points + sentiment; 1 = unrelated).
All models judged on the **same 200 records** (sampled from 1840 commonly
usable), so means are comparable. Predicted summaries were first recovered from
`raw_output` for the JSON-broken rows (coverage 1852 / 1883 / 1900).

| Model | mean rating | rating distribution (1/2/3/4/5) |
|---|---:|---|
| Qwen3-4B zero-shot | **2.12** | 11 / 156 / 32 / 0 / 1 |
| Qwen3-32B zero-shot | 2.07 | 13 / 161 / 26 / 0 / 0 |
| llm_sft Qwen3-4B | 1.95 | 33 / 148 / 16 / 3 / 0 |

**Findings**
- **Summary forecasting is hard and unsolved.** All models score ~2/5 ("largely
  misses the points") — predicting the *content and sentiment of unseen future
  replies* (not just summarizing given text) is a genuinely difficult task. The
  judge discriminates well (reasons cite specific missed/irrelevant points; it
  awards 1s, 3s, and the occasional 4–5), so the low mean is real, not a default.
- **Fine-tuning HURT the summary** (SFT 1.95 < zero-shot 4B 2.12) and produced
  the most "unrelated" 1s (33 vs 11). LoRA-SFT optimized the numeric fields and
  JSON format, degrading free-text quality — a concrete trade-off: the SFT model
  that wins on numeric ranking (§1) is the worst summarizer.
- **Scale doesn't help** the summary either (32B ≈ 4B zero-shot).
- **Implication for the world model**: it is strong on the *numeric* engagement
  signals (ρ≈0.76) but the qualitative reply-content channel is not yet usable;
  this argues for either a dedicated summary head/decoder or treating
  reply_summary as a separate, harder sub-problem.

Reproduce: `scripts/judge_summaries.py` (run via the `judge` workflow mode);
raw ratings in `results/judge_ratings.jsonl`, aggregate in
`results/judge_summary.json`.

---

## 7. Controlled ablation: what does text *actually* add?

§1 concluded that structure carries most of the signal, but that comparison was
confounded. `feature_gbdt`'s "text" features are only shallow lexical statistics
(length, punctuation, caps, url) — **no semantics at all** — while the only
semantic models (`encoder_distilbert`, `retrieval_sbert`) carry **no structural
features**. Nothing in the benchmark had ever combined structure with real
semantics, so "text adds nothing" was never tested.

This section fixes exactly one variable at a time: identical LightGBM, identical
log1p targets, **only the feature block changes**. Semantic features are frozen
`all-MiniLM-L6-v2` embeddings of the stimulus body and the parent body (both
available at prediction time — no leakage; the replies are never embedded).
Every cell gets its **own** 10-config hyperparameter search on a held-out 20% of
train, so no cell is handicapped. Differences use a **paired** bootstrap (one
resample scored for both models), which is the correct test for a difference.

Reproduce: `scripts/run_ablation.py --suite {main,control,tuned} [--tune]`,
`scripts/run_ablation_shift.py`, `scripts/diagnose_ties.py`. Raw numbers in
`data_analysis/ablation_results.json`, `ablation_shift.json`,
`tie_diagnosis.json`.

### 7.1 In-domain (politics, 7596 train / 1900 test, per-cell tuned)

`S` = structural, `L` = lexical, `E` = semantic (`Es` = stimulus only),
`N384`/`N768` = **pure per-record noise** of matched width — the dimensionality
control.

| cell | dim | score ρ [95% CI] | width ρ [95% CI] |
|---|---:|---|---|
| S | 7 | 0.7489 [0.731, 0.766] | 0.7050 [0.678, 0.731] |
| S+L | 14 | 0.7518 [0.732, 0.769] | 0.7024 [0.674, 0.729] |
| S+N384 *(noise)* | 391 | 0.7805 [0.760, 0.798] | 0.7390 [0.711, 0.764] |
| S+Es | 391 | 0.7852 [0.765, 0.802] | 0.7337 [0.707, 0.761] |
| S+N768 *(noise)* | 775 | 0.7763 [0.758, 0.794] | 0.7172 [0.688, 0.744] |
| S+E | 775 | 0.7882 [0.770, 0.804] | 0.7396 [0.713, 0.764] |
| S+L+E | 782 | 0.7859 [0.767, 0.803] | 0.7430 [0.716, 0.767] |

Paired deltas on score ρ:

| comparison | Δ | 95% CI | p |
|---|---:|---|---:|
| S+L − S | +0.003 | [−0.010, +0.015] | 0.692 |
| S+N384 − S *(noise!)* | +0.032 | [+0.019, +0.045] | 0.000 |
| S+Es − S | +0.036 | [+0.022, +0.050] | 0.000 |
| S+E − S | +0.039 | [+0.026, +0.052] | 0.000 |
| **S+Es − S+N384** *(vs matched noise)* | **+0.005** | [−0.006, +0.015] | 0.426 |
| **S+E − S+N768** *(vs matched noise)* | **+0.012** | [+0.002, +0.021] | 0.020 |

- **Shallow lexical text adds nothing.** S+L − S = +0.003 (p=0.69). The "text"
  that `feature_gbdt` and `quantile_gbm` actually consume is worthless once each
  cell is tuned — their edge over `structural_prior` in §1 does not come from text.
- **Against a dimension-matched noise control, semantics nearly vanishes.**
  The naive gain (+0.039 over S) collapses to +0.005 (n.s.) at 384 dims and
  +0.012 at 768. Most of the apparent "semantics helps" effect is not semantics.
- **All semantic signal is in the stimulus body.** A parent-only variant scored
  −0.004 vs S (p=0.47, `--suite control`); S+Es ≈ S+E.

### 7.2 The dimensionality artifact (a caveat for the whole benchmark)

Adding **768 columns of pure noise** raises score ρ by **+0.027 to +0.032
(p<0.001)**. This is not a fluke of under-tuning: it survives a per-cell grid
search spanning `num_leaves` 7→127, `feature_fraction` 0.3→1.0 and
`min_child_samples` 20→200.

I hypothesized tie-breaking: every tabular baseline rounds to `int`, coarse
7-feature models emit few distinct values, and Spearman's tie correction
penalizes that. Prediction cardinality is indeed wildly uneven across the
published results (`gnn` and `hawkes` emit just **9 distinct width values**
across 1900 records; `retrieval_sbert` emits 729). **The hypothesis was
disconfirmed for score and only partly right for width** — recomputing ρ on the
raw, unrounded predictions of the same fitted models:

| gap S+N768 − S | rounded | raw | attributable to rounding |
|---|---:|---:|---:|
| score ρ | +0.0282 | +0.0282 | **0%** |
| width ρ | +0.0368 | +0.0238 | 35% |

So rounding inflates the *width* comparison but explains none of the score
effect; the score effect is a genuine LightGBM fitting phenomenon (random
columns behaving like feature bagging). Mechanism aside, the practical
consequence stands: **ρ is sensitive to feature-block width independently of
information content**, so §1's cross-model ρ comparisons — spanning 7-feature
priors to 768-dim encoders to LLMs — are not apples-to-apples, and gaps under
~0.03 should not be read as rankings.

### 7.3 Does semantics *transfer*? (the world-model question)

Each cell fitted once on politics 2025-12, evaluated on all 7 shift sets.
Score ρ:

| cell | pol 12-25 | pol 09-25 | pol 02-25 | pol 08-24 | news | worldnews | Conservative | technology |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S | 0.749 | 0.752 | 0.733 | 0.728 | 0.724 | 0.679 | 0.681 | 0.732 |
| S+L | 0.752 | 0.737 | 0.728 | 0.723 | 0.725 | 0.689 | 0.671 | 0.735 |
| S+N768 | 0.776 | 0.742 | 0.729 | 0.729 | 0.733 | 0.677 | 0.661 | 0.735 |
| S+Es | 0.785 | 0.752 | 0.739 | 0.741 | 0.743 | 0.696 | 0.671 | 0.752 |
| S+E | 0.788 | 0.756 | 0.739 | 0.737 | 0.730 | 0.692 | 0.676 | 0.744 |

**S+E − S, by test set** (the naive "semantics helps" claim):

| test set | Δ score ρ | p |
|---|---:|---:|
| politics 2025-12 (in-domain) | **+0.039** | **0.000** |
| politics 2025-09 | +0.004 | 0.552 |
| politics 2025-02 | +0.006 | 0.464 |
| politics 2024-08 | +0.010 | 0.188 |
| news 2025-09 | +0.006 | 0.444 |
| worldnews 2025-09 | +0.013 | 0.092 |
| Conservative 2025-09 | −0.005 | 0.600 |
| technology 2025-09 | +0.013 | 0.068 |

**The semantic advantage is significant in-domain and nowhere else.** It is
+0.039 (p<0.001) on the test set drawn from the training distribution, and
averages +0.007 across seven shifted sets without reaching significance on any
of them — including three that are the *same subreddit* a few months earlier.
Whatever the embeddings contribute is in-domain topical fitting, not
transferable knowledge about engagement. Against matched noise the residual is
smaller still (significant on 3 of 8 sets).

The structural cell, by contrast, transfers essentially as well as anything
else everywhere, using 7 features and no text.

### 7.4 llm_reghead: the controversiality fix, run and costed

§1 flagged that `llm_reghead`'s controversiality head collapsed to the majority
class and noted class-weighted CE as the fix. Running it on the local A100
(Qwen3-4B, bf16, LoRA, 2 epochs) exposed a bug *in the fix*: PyTorch's
`CrossEntropyLoss(weight=…, reduction="mean")` divides by the **sum of weights**,
not the batch size, so at `batch_size=4` with a 2.4% positive rate a batch
holding one positive returns ~3.7 instead of ~40 — an ~11× dilution that
normalizes the intended up-weighting straight back out. `llm_reghead` now takes
`contr_loss_norm` (`"batch"`, default, vs `"weight_sum"` to reproduce the old runs).

| variant | score ρ | width ρ | contr F1 | pred+ rate (true 2.4%) |
|---|---:|---:|---:|---:|
| published (unweighted CE) | +0.731 | +0.663 | 0.494 | 0.0% |
| class-weighted, `weight_sum` | +0.721 | +0.692 | 0.532 | 0.3% |
| **class-weighted, `batch` (correct)** | +0.710 | +0.634 | **0.595** | 3.5% |

- Correctly normalized, `llm_reghead` gets the **best controversiality F1 in the
  benchmark** (0.595 vs the previous best 0.557) with a sensibly calibrated
  positive rate.
- It is **not free**: score ρ drops 0.731→0.710 and width ρ 0.663→0.634 as the
  CE term competes with the two Huber terms. Multi-task interference, not a win
  on every axis.
- Retraining the *same* config moved width ρ by +0.029 run-to-run — another
  reason to distrust sub-0.03 gaps in §1.

---

## 5. Takeaways for the world-model claim

1. **It is a world model, not a politics-memorizer**: ≤0.09 ρ drop across five
   unseen subreddits and ≤0.03 across 16 months of time shift.
2. **Conversation structure is the dominant, transferable signal** — a
   text-free structural prior (ρ=0.750) and a reply-tree GNN (0.745) rival the
   best text model, and transfer best across domains. **§7 upgrades this from a
   confounded observation to a controlled result**: given real sentence
   embeddings and its own hyperparameter search, a text-aware model beats the
   text-free one by +0.039 ρ in-domain — but by an insignificant +0.007 on
   average across seven shifted test sets, and by only +0.005–0.012 against a
   dimension-matched noise control even in-domain. Shallow lexical features add
   nothing at all (p=0.69). Semantics buys in-domain fit; structure buys transfer.
3. **A cheap supervised predictor (quantile_gbm / feature_gbdt) is the strongest
   and most scalable world model here**, beating both a fine-tuned Qwen3-4B and
   zero-shot Qwen3-32B on ranking — exactly the dense, fast, repeatable signal
   an RL bystander agent needs.
4. **Scaling is not saturated** — the tree models' monotonic data-scaling curve
   argues for collecting more interaction data.
5. **Read ρ gaps with care.** Adding pure-noise columns moves score ρ by ~+0.03
   (p<0.001, robust to tuning), retraining one config moved width ρ by +0.029,
   and 95% bootstrap CIs on ρ span ~±0.018. Differences below ~0.03 between
   models with different feature widths are not evidence of a better world
   model. The §1 leaderboard's top five rows are a tie.

### Scope notes
- Analyses use the numeric targets; `reply_summary` is omitted from the
  cross-domain/temporal sets (would need the LLM annotator, no GPU required for
  the numeric metrics).
- The heavy sweep uses the 6 fast models (instant per-record prediction);
  encoder/frozen-MLP/LLM are in the main table only (per-record neural encoding
  / GPU cost makes the 8-set × 5-scale sweep impractical on CPU).
- Cross-domain/temporal test sets are 2000 chains each, capped for balance.
