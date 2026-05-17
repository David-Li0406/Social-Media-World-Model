import json
import argparse
from pathlib import Path
from json.decoder import JSONDecodeError
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import os

# ---------------------------------------------------------------------------
# Suggested models for baseline testing (all fit on a single H100 80GB):
#
#   "Qwen/Qwen3-4B"                          -- current (~4B, ~2GB 4-bit)
#   "meta-llama/Llama-3.1-8B-Instruct"       -- strong 8B instruction baseline
#   "Qwen/Qwen3-32B"                          -- mid-range (~32B, ~16GB 4-bit)
#   "meta-llama/Llama-3.3-70B-Instruct"      -- large (~70B, ~35GB 4-bit)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HuggingFace model ID")
parser.add_argument("--input", default="train_test/RC_2025-12_politics_width_chain01_test.jsonl")
parser.add_argument("--max-new-tokens", type=int, default=500)
parser.add_argument("--retries", type=int, default=3)
args = parser.parse_args()

hugging_face_token = os.getenv('HUGGING_FACE')

login(hugging_face_token)

assert torch.cuda.is_available(), "No CUDA device found"
print(f"Using device: {torch.cuda.get_device_name(0)}")

model_id = args.model
model_slug = model_id.replace("/", "_")

tokenizer = AutoTokenizer.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda:0",
)


def generate_json_response(prompt_text, max_new_tokens=500, retries=3):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    messages = [{"role": "user", "content": prompt_text}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    for attempt in range(retries):
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        raw = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        try:
            return json.loads(raw), raw
        except JSONDecodeError:
            if attempt < retries - 1:
                print(f"  JSON parse failed (attempt {attempt + 1}), retrying...")
            else:
                print("  All attempts failed — storing raw output.")
                return None, raw


input_path = Path(args.input)
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
output_path = results_dir / f"{model_slug}_results.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

print(f"Running inference on {len(records)} test records with {model_id}")
print(f"Output: {output_path}\n")

with open(output_path, "w", encoding="utf-8") as out_f:
    for i, record in enumerate(records):
        predicted, raw = generate_json_response(
            record["prompt"],
            max_new_tokens=args.max_new_tokens,
            retries=args.retries,
        )

        result = {
            "record_id": record["record_id"],
            "predicted": predicted,
            "ground_truth": json.loads(record["completion"]),
            "raw_output": raw,
        }

        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()

        status = "OK" if predicted is not None else "FAIL"
        print(f"[{i+1}/{len(records)}] {status}")

print(f"\nDone. Results written to {output_path}")
