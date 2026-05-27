#!/usr/bin/env bash
# Verify (and lazily prepare) the dataset on Machine B.
# Idempotent: skips when DONE_FILE matches DATA_VERSION.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/datasets/smwm}"
DATA_VERSION="${DATA_VERSION:-v1}"
DONE_FILE="$DATA_ROOT/.prepared-$DATA_VERSION"

mkdir -p "$DATA_ROOT"

if [ -f "$DONE_FILE" ]; then
  echo "[prepare_data] already prepared at $DATA_ROOT (version $DATA_VERSION)"
  exit 0
fi

echo "[prepare_data] preparing dataset at $DATA_ROOT (version $DATA_VERSION)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The committed train/test JSONL files (~42 MB total) are the source of truth.
# We just copy them into DATA_ROOT so DeepSpeed / training scripts can point at
# a stable location independent of the workspace.
for f in \
  "train_test/RC_2025-12_politics_width_chain01_train.jsonl" \
  "train_test/RC_2025-12_politics_width_chain01_test.jsonl"
do
  src="$REPO_ROOT/$f"
  if [ ! -f "$src" ]; then
    echo "[prepare_data] missing $src — aborting" >&2
    exit 1
  fi
  dst="$DATA_ROOT/$(basename "$f")"
  cp -u "$src" "$dst"
  echo "[prepare_data] $(basename "$f"): $(wc -l <"$dst") lines"
done

# A small debug subset for fast smoke-test training (first 200 records).
head -n 160 "$DATA_ROOT/RC_2025-12_politics_width_chain01_train.jsonl" > "$DATA_ROOT/train_debug.jsonl"
head -n 40  "$DATA_ROOT/RC_2025-12_politics_width_chain01_test.jsonl"  > "$DATA_ROOT/test_debug.jsonl"
echo "[prepare_data] debug subsets: train=160 test=40"

rm -f "$DATA_ROOT"/.prepared-*
touch "$DONE_FILE"
echo "[prepare_data] done -> $DONE_FILE"
