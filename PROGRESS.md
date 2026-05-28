# Social Media World Model — Progress Report

**Date:** 2026-05-18
**Author:** Dawei Li (daweili5@asu.edu)
**Repo:** `social_media_world_simulation` (branch: `main`)

---

## 1. Project goal

Build a **social media world model** — a predictor that, given a Reddit reply
chain and a stimulus comment, outputs the engagement signals the stimulus is
likely to receive:

- `score` — net upvotes (integer, heavy-tailed)
- `controversiality` — 0/1 flag
- `width` — number of direct reply comments
- `reply_summary` — text summary of those replies

The motivation (per the proposal *Social Media World Model for Bystander Agent
Training*) is to provide a fast, dense substitute for full LLM-based social
simulation so that downstream RL bystander agents can be trained without
real-platform feedback loops.

---

## 2. Starting state (before this work)

The repo was a flat collection of 8 scripts in the root with hardcoded paths:

```
buildTree.py        # parquet -> per-post comment forest
buildChain.py       # forest -> deduplicated reply chains  (called buildTree.py in code)
buildChainSets.py   # chains -> (context, stimulus, ground_truth) samples
findSamples.py      # LLM summarizes reply bodies; collapses width-list -> count
buildTrainTest.py   # train/test split as prompt+completion JSONL
fineTune.py         # QLoRA SFT of Qwen3-4B
inference.py        # zero-shot/SFT inference, JSON output
metrics.py          # Score MSE, Width MSE, Controversiality macro-F1
```

Plus two SLURM wrappers (`submit_tests.sh`, `submit_metrics.sh`) that hardcode
four LLM model IDs:

- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-32B`
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`

**Existing result files** (under `results/`):

- `Qwen_Qwen3-4B_results.jsonl` (1900 test records)
- `Qwen_Qwen3-32B_results.jsonl` (801 valid, 1099 parse failures)

**Pain points identified:**

- All paths hardcoded → can't run on a new dataset without editing 5+ files.
- No `Baseline` abstraction → every new method needs edits to
  `inference.py`, `submit_tests.sh`, and `metrics.py`.
- Only point-wise metrics (MSE, F1). No measure of whether the model gets the
  **relative order** of items right — critical because Reddit scores are
  heavy-tailed and absolute-error metrics are dominated by viral outliers.

---

## 3. What was delivered this iteration

### 3a. Package restructure → `src/smwm/`

```
src/smwm/
  config.py              # YAML loader
  data/
    io.py                # JSON/JSONL helpers + shared schema utilities
    build_tree.py        # ported from buildTree.py
    build_chains.py      # ported from buildChain.py
    build_chain_sets.py  # ported from buildChainSets.py
    summarize_replies.py # ported from findSamples.py (lazy GPU imports)
    build_splits.py      # ported from buildTrainTest.py (now emits
                         #   context/stimulus/ground_truth alongside prompt/completion
                         #   so non-LLM baselines can read structured fields)
  baselines/
    base.py              # Baseline ABC + record-field accessors
    registry.py          # @register decorator + name -> class lookup
    constant.py          # constant_mean, constant_median
    feature.py           # feature_gbdt (GradientBoostingRegressor + LogReg)
    retrieval.py         # retrieval_tfidf, retrieval_sbert
    encoder.py           # encoder (distilbert / deberta multi-head)
    llm.py               # llm wrapper for any HF causal LM (zero-shot)
  metrics/
    pointwise.py         # mse, mae, f1_binary_macro  (ported)
    ranking.py           # spearman, pairwise_accuracy  (new — see §3c)
    evaluate.py          # results.jsonl -> dict of metric -> value
  cli/
    run_data.py          # python -m smwm.cli.run_data --stage tree|chains|sets|summarize|splits|all
    run_inference.py     # python -m smwm.cli.run_inference --config configs/baseline/X.yaml
    run_eval.py          # python -m smwm.cli.run_eval results/*.jsonl
configs/
  data.yaml              # paths, seed, test_fraction, min_root_score
  data_smoke.yaml        # 200-record subset for smoke tests
  eval.yaml              # which metrics to compute per target
  baseline/
    constant_mean.yaml, constant_median.yaml
    feature_gbdt.yaml
    retrieval_tfidf.yaml, retrieval_sbert.yaml
    encoder_distilbert.yaml, encoder_deberta.yaml
    llm_qwen3_4b.yaml, llm_qwen3_32b.yaml
    llm_llama31_8b.yaml, llm_llama33_70b.yaml
scripts/slurm/
  submit_baselines.sh    # loops over baseline configs; auto-adds GPU for encoder/llm
  submit_eval.sh         # evaluates results/*.jsonl
tests/
  test_metrics_ranking.py    # Spearman / pairwise_accuracy sanity checks
  test_baseline_registry.py  # all baselines registered + return correct dict shape
pyproject.toml
PROGRESS.md (this file)
```

All four original target fields (`score`, `controversiality`, `width`,
`reply_summary`) are preserved end-to-end. The flat root scripts are left in
place untouched for backwards compatibility.

### 3b. New baselines

Seven baselines now share a single interface:

```python
class Baseline(ABC):
    def fit(self, train_records: list[dict]) -> None: ...
    def predict(self, record: dict) -> dict   # -> {score, controversiality, width, reply_summary}
```

| Name | Type | Targets | Notes |
|---|---|---|---|
| `constant_mean` | floor | score (mean), width (mean), contr (majority) | sanity baseline; constant predictions → Spearman undefined |
| `constant_median` | floor | score/width median, contr majority | robust to outliers |
| `feature_gbdt` | metadata | GBDT on log1p(score), log1p(width); LogReg on controversiality | 13 features: body length, punctuation, parent/root score, depth, hour-of-day, etc. |
| `retrieval_tfidf` | retrieval | similarity-weighted average of top-k train neighbours | TF-IDF on `context + [SEP] + stimulus.body` |
| `retrieval_sbert` | retrieval | same | `all-MiniLM-L6-v2` sentence embeddings, CPU by default |
| `encoder` | neural | DistilBERT or DeBERTa multi-head | 2 regression heads (log1p) + 1 binary classifier; joint training with Huber + CE |
| `llm` | zero-shot LLM | parses model JSON output | wraps any HF causal LM (Qwen, Llama, …) |

The four original LLM model IDs are now four YAMLs under
`configs/baseline/llm_*.yaml` — no code change needed to add a new LLM, just a
new YAML.

### 3c. Ranking-based metric (§ the third deliverable)

The user's example: if truth is `(10, 20)` and prediction is `(100, 200)`, the
relative order is preserved and the model should get full credit — even
though MSE is enormous.

Added two ranking metrics in `src/smwm/metrics/ranking.py`:

- **Spearman ρ** (`scipy.stats.spearmanr`): rank correlation in [-1, 1].
  ρ=1.0 for the example above, ρ=-1.0 for reversed, ρ≈0 for shuffled.
  Returns NaN when predictions or truths are constant (handled explicitly).
- **Pairwise accuracy**: fraction of test pairs `(i,j)` with
  `sign(pred_i − pred_j) == sign(true_i − true_j)`. Tied truths are excluded
  from the denominator; tied predictions on non-tied truths count as wrong
  (so constant baselines correctly score 0).

Both are applied to `score` and `width`. Skipped for `controversiality`
(binary — F1 is already the right metric).

`run_eval.py` prints them alongside MSE/MAE/F1 in one table.

---

## 4. End-to-end smoke test

Setup: 200-record subset of `sim_output/RC_2025-12_politics_width_chain01.jsonl`,
split 160 train / 40 test (`configs/data_smoke.yaml`).

All six non-LLM baselines fitted and predicted successfully through the new
CLI. Results:

```
============================================================================
Model                MSE-score    Spearman ρ   pair_acc     MSE-width   ρ-width
============================================================================
constant_mean         857,694         NaN        0.000        116.75      NaN
constant_median       903,040         NaN        0.000        123.05      NaN
retrieval_tfidf     1,775,802        +0.186      0.560        510.53     +0.150
retrieval_sbert       872,871        +0.512      0.687        270.03     +0.366
encoder_distilbert    898,326        +0.385      0.612        116.10     +0.274
feature_gbdt          361,048        +0.730      0.765         59.35     +0.581   <-- best
============================================================================
```

For comparison, evaluating the **pre-existing** LLM result files with the new
ranking metric (no re-run required):

```
Model                  N        score MSE     score ρ    score pair_acc    width ρ
Qwen3-4B (zero-shot)   1900    2,427,835    +0.374       0.573             +0.508
Qwen3-32B (zero-shot)   801    3,404,479    +0.471       0.647             +0.491
```

**Observations:**
- Feature-based GBDT is the strongest non-LLM baseline on this small sample
  (ρ=0.73 on score, 0.58 on width).
- Retrieval-SBERT beats retrieval-TF-IDF substantially on ranking, confirming
  that semantic similarity matters more than lexical overlap.
- Zero-shot LLMs (Qwen3-32B) achieve ρ=0.47 on score over the *full* 1900-record
  test set — competitive but not obviously better than a feature GBDT trained
  on the same data. Full-data GBDT comparison is the immediate next experiment.
- All "constant" baselines correctly produce NaN Spearman / 0 pairwise
  accuracy, validating the metric implementation.

### Unit tests
`pytest -q tests/` → **6 passed in 5.3s**. Covers Spearman edge cases
(perfect-scale, reversed) and registry/shape contracts.

---

## 4b. Full-data benchmark incl. fine-tuned Qwen3-4B (LoRA SFT)

All baselines trained on the full 7596 / evaluated on the full 1900 test set.
The `llm_sft` row is Qwen3-4B LoRA-fine-tuned via the `remote-gpu-exp` GitHub
Actions workflow on the self-hosted H20 runner (transformers+PEFT backend,
2 epochs, rank-16 LoRA, ~1h52m train; inference ~2.5h, max_new_tokens 256).

```
Model                  score MSE    score ρ   score pair_acc   width MSE  width ρ   contr F1
constant_mean          1,983,886    nan       0.000            186.4      nan       0.494
constant_median        2,127,061    nan       0.000            201.8      nan       0.494
retrieval_tfidf        1,618,963    +0.274    0.597            198.1      +0.215    0.514
retrieval_sbert        1,585,864    +0.238    0.583            191.5      +0.201    0.514
Qwen3-4B zero-shot     2,427,835    +0.374    0.573            563.4      +0.508    0.223
Qwen3-32B zero-shot*   3,404,479    +0.471    0.647            632.9      +0.491    0.302
encoder_distilbert     1,313,780    +0.720    0.761            118.7      +0.642    0.494
llm_sft Qwen3-4B       2,068,522    +0.724    0.742            182.3      +0.541    0.545   <-- fine-tuned
feature_gbdt           1,906,139    +0.759    0.770            147.0      +0.676    0.424   <-- best ranking
```
*Qwen3-32B scored on 801 records (1099 un-parseable JSON outputs).

Key takeaways:
- **Fine-tuning massively helps the LLM**: SFT lifts Qwen3-4B score ρ from
  0.374 → **0.724**, width ρ 0.508 → 0.541, controversiality F1 0.223 →
  **0.545** (best F1 of all methods), and slashes width MAE 13.6 → 4.45. The
  tuned 4B even beats zero-shot Qwen3-**32B** on every metric.
- It reaches the encoder's ranking quality (ρ≈0.72) but a sub-second
  `feature_gbdt` still edges it on ranking (score ρ 0.759, width ρ 0.676) at a
  tiny fraction of the cost — the headline finding stands.
- **Parsing caveat**: the SFT model emits valid numbers but occasionally a
  malformed JSON tail (a stray quote); 81% of raw outputs failed strict
  `json.loads`. `llm_sft.predict()` now has a regex fallback that recovers the
  numeric fields, so reported numbers reflect the model's real predictions.

---

## 5. How to run

```bash
# 1. Generate splits from a sim_output JSONL (uses the existing 9.5k-record file)
PYTHONPATH=src python3 -m smwm.cli.run_data --stage splits --config configs/data.yaml

# 2. Run any baseline
PYTHONPATH=src python3 -m smwm.cli.run_inference \
    --config configs/baseline/feature_gbdt.yaml \
    --train  train_test/RC_2025-12_politics_width_chain01_train.jsonl \
    --test   train_test/RC_2025-12_politics_width_chain01_test.jsonl

# 3. Evaluate one or many result files
PYTHONPATH=src python3 -m smwm.cli.run_eval results/*.jsonl

# 4. Cluster: submit one job per baseline (auto-allocates GPU for encoder/llm)
scripts/slurm/submit_baselines.sh                                # all configs
scripts/slurm/submit_baselines.sh configs/baseline/llm_qwen3_4b.yaml   # one config

# Adding a new baseline: 1 class with @register("name") + 1 YAML, zero edits elsewhere.
```

---

## 6. What's still pending

| # | Item | Effort | Blocking? |
|---|---|---|---|
| 1 | **Run `llm` baseline end-to-end on real GPU** to verify the wrapper reproduces the existing `results/Qwen_Qwen3-4B_results.jsonl` byte-for-byte. Code is written, registered, configured — just needs a SLURM submission. | 1 GPU job | No |
| 2 | **Port `fineTune.py`** to `baselines/llm_sft.py` so the QLoRA-tuned model becomes another registered baseline. Original `fineTune.py` still works as-is. | ½ day | No |
| 3 | **Full-data benchmark**: run every baseline on the full 7596/1900 train/test split and produce a comparison table including the existing LLM numbers. | 1–2 GPU days | Yes — this is what tells us which method to invest in. |
| 4 | **Better `reply_summary` evaluation**: today only LLM/retrieval baselines emit text; metrics ignore the text field. Consider ROUGE-L or embedding cosine vs ground truth. | 1 day | No, but worth deciding before publication. |
| 5 | **Encoder reply-summary head**: currently leaves it blank. Could add a small T5/BART decoder, or punt and let the LLM remain authoritative for text. | 1–2 days | No |
| 6 | **Deprecation shims** for root-level scripts (`buildTree.py`, `inference.py`, …) that forward to the new CLI. | 1 hour | No |
| 7 | **Larger/different data domains** (other subreddits, different time windows) to validate the world-model claim of generalisation. | varies | Yes for the longer-term proposal |

---

## 7. Recommended next step

Run **item #3 — full-data benchmark** — by submitting the existing
`scripts/slurm/submit_baselines.sh` over all configs except the giant LLMs,
then evaluate with `submit_eval.sh`. This will give the first proper
apples-to-apples comparison of cheap baselines vs the LLM zero-shot numbers
already on disk, and will tell us whether the world-model formulation is best
served by a sub-second feature regressor or a multi-billion-parameter LLM.
