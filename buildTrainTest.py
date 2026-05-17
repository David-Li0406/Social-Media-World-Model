import json
import random
from pathlib import Path

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

INPUT_PATH = Path("sim_output/RC_2025-12_politics_width_chain01.jsonl")
TRAIN_OUT = Path("train_test/RC_2025-12_politics_width_chain01_train.jsonl")
TEST_OUT = Path("train_test/RC_2025-12_politics_width_chain01_test.jsonl")
TEST_FRACTION = 0.2


def make_prompt(context, stimulus):
    return (
        "You are analyzing a Reddit political discussion. "
        "Given the conversation context and a stimulus comment, predict:\n"
        "  1. The score (upvotes) the stimulus will receive\n"
        "  2. Whether the stimulus is controversial (0 or 1)\n"
        "  3. The number of direct reply comments (width) the stimulus will generate\n\n"
        "  4. A summary of the body fields of the direct reply comments (width) the stimulus will generate"
        "Return a JSON object with exactly these fields:\n"
        '  - "score": an integer with the predicted score\n'
        '  - "controversiality": 0 or 1\n'
        '  - "width": an integer with the predicted number of direct reply comments'
        '  - "reply_summary": a string'
        "Do NOT include anything outside the JSON object.\n\n"
        "conversation context:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        "stimulus comment:\n"
        f"{json.dumps(stimulus, ensure_ascii=False, indent=2)}\n\n"
        "Now provide score, controversiality, width, and reply_summary as a JSON object."
    )


def make_target(ground_truth):
    return json.dumps(ground_truth, ensure_ascii=False)


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

examples = []

for idx, record in enumerate(records):
    context = record.get("context", [])
    stimulus = record.get("stimulus")
    ground_truth = record.get("ground_truth")

    if stimulus is None or ground_truth is None:
        continue
    if ground_truth.get("score") is None or ground_truth.get("reply_summary") is None:
        continue

    examples.append(
        {
            "record_id": idx,
            "prompt": make_prompt(context, stimulus),
            "completion": make_target(ground_truth),
        }
    )

random.shuffle(examples)

split_idx = int(len(examples) * (1 - TEST_FRACTION))
train_examples = examples[:split_idx]
test_examples = examples[split_idx:]

for path, rows in [(TRAIN_OUT, train_examples), (TEST_OUT, test_examples)]:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"total examples: {len(examples)}")
print(f"train: {len(train_examples)}")
print(f"test:  {len(test_examples)}")
