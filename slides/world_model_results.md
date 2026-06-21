# Social Media World Model
### Benchmark, Generalization & Reply-Summary Evaluation

- **Task**: predict a stimulus comment's engagement — score, controversiality, reply width, reply summary
- **Headline metric**: Spearman ρ (scale-invariant ranking) + pairwise accuracy; macro-F1 for controversiality
- **What's new**: 18 baselines (was 6) · cross-domain · temporal · data-scaling · LLM-as-judge on summaries
- **Data**: Reddit r/politics (7,596 train / 1,900 test) + 5 subreddits × 3 months for generalization
- Qwen3-4B LoRA fine-tuned on H20 GPUs via the GitHub Actions runner

---

## 1 · Main Benchmark (r/politics, 1,900 test)
*Cheap supervised models lead; conversation structure carries most of the signal*

| Model | score ρ | width ρ | contr F1 | type |
|---|---:|---:|---:|---|
| **quantile_gbm (LightGBM)** | **0.760** | **0.687** | **0.557** | tabular |
| feature_gbdt | 0.759 | 0.676 | 0.424 | tabular |
| structural_prior (NO text) | 0.750 | 0.670 | 0.415 | graph-only |
| gnn (reply-tree) | 0.745 | 0.676 | 0.494 | graph NN |
| llm_reghead Qwen3-4B | 0.731 | 0.663 | 0.494 | LLM + heads |
| llm_sft Qwen3-4B (JSON) | 0.724 | 0.541 | 0.545 | LLM SFT |
| encoder (DistilBERT) | 0.720 | 0.642 | 0.494 | encoder |
| Qwen3-32B zero-shot | 0.482 | 0.477 | 0.311 | LLM 0-shot |
| Qwen3-4B zero-shot | 0.374 | 0.508 | 0.223 | LLM 0-shot |

**Key findings**
- **Structure > text** — a text-free structural prior (ρ=0.750) and reply-tree GNN (0.745) rival the best text model.
- **Cheap wins** — a sub-second LightGBM beats every LLM on ranking.
- **Reg-head > JSON** — `llm_reghead` lifts SFT 0.724→0.731 (score), 0.541→0.663 (width), **0% parse failures vs 81%**.
- **Fine-tuning helps LLMs a lot** — Qwen3-4B 0.374→0.724; tuned 4B beats zero-shot 32B.

---

## 2 · Generalization: is it a world model?
*Trains on politics, transfers across domains & time; keeps improving with data*

**Cross-domain** (score ρ, train = politics) | **Temporal** (politics) | **Data-scaling** (score ρ)

| test domain | qgbm | gnn |   | gap | qgbm | feat |   | train n | qgbm | gnn |
|---|---:|---:|---|---|---:|---:|---|---|---:|---:|
| politics (in-dom) | 0.760 | 0.746 |   | 0 mo | 0.760 | 0.759 |   | 500 | 0.698 | 0.710 |
| technology | 0.737 | 0.752 |   | −3 mo | 0.745 | 0.750 |   | 1,000 | 0.723 | 0.737 |
| news | 0.723 | 0.736 |   | −10 mo | 0.738 | 0.741 |   | 2,000 | 0.731 | 0.739 |
| worldnews | 0.685 | 0.695 |   | −16 mo | 0.731 | 0.736 |   | 4,000 | 0.743 | 0.740 |
| Conservative | 0.670 | 0.685 |   |  |  |  |   | 7,596 | 0.760 | 0.742 |

**Takeaways**
- **Cross-domain**: graceful — only ≈0.02–0.09 ρ drop to 4 unseen subreddits; structural/GNN transfer best.
- **Temporal**: very stable — ≤0.03 ρ over 16 months back; engagement dynamics don't rot. Time-shift < domain-shift.
- **Scaling**: tree models monotonic & **not saturated** at 7,596 → more data should help; linear GLMs saturate early.

---

## 3 · Reply-Summary Quality — LLM-as-Judge (Qwen3-32B)
*200 common records rated 1–5 vs the gold reference summary*

| Model | mean | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| **Qwen3-4B zero-shot** | **2.12** | 11 | 156 | 32 | 0 | 1 |
| Qwen3-32B zero-shot | 2.07 | 13 | 161 | 26 | 0 | 0 |
| llm_sft Qwen3-4B | 1.95 | 33 | 148 | 16 | 3 | 0 |

**Findings**
- **Summary forecasting is unsolved** — all ~2/5: predicting the content/sentiment of *unseen future replies* is genuinely hard.
- **Fine-tuning HURT it** — SFT is worst (1.95) with the most "unrelated" 1s; LoRA-SFT traded text quality for numeric/JSON.
- **Scale doesn't help** — 32B ≈ 4B zero-shot.
- **Judge is reliable** — discriminates 1/3/4–5 with specific reasons; 0% parse failures.

**Overall conclusions**
1. Conversation **structure** is the dominant, domain/time-transferable signal.
2. A cheap LightGBM is the strongest, most scalable world model — beats fine-tuned & 32B LLMs on ranking.
3. Numeric channel is strong (ρ≈0.76) and generalizes; **text (reply summary) channel is not yet usable**.
4. Numeric-vs-text trade-off: the SFT model best on numbers is the worst summarizer.
5. Next: dedicated summary decoder; imbalance-aware controversiality; scale the dataset.
