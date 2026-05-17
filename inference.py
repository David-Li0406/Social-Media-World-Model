import json
import random
from pathlib import Path
from json.decoder import JSONDecodeError
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import os

hugging_face_token = os.getenv('HUGGING_FACE')

login(hugging_face_token)

# ---------- GPU check ----------
assert torch.cuda.is_available(), "No CUDA device found"
device_name = torch.cuda.get_device_name(0)
print(f"Using device: {device_name}")

model_id = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda:0"
)

def makeMessage(replies: list[dict]) -> list[dict]:
    bodies = "\n\n".join(
        f"- {reply['body']}" for reply in replies
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a summarization assistant. "
                "Given a list of replies to a social media comment, summarize the key points and overall sentiment expressed across the body section of all replies. "
                "Return a JSON object with exactly one field:\n"
                '  - "summary": a string containing the summary.\n\n'
                "Do NOT include anything outside the JSON object. "
                "Do not add extra keys, text, or formatting."
            ),
        },
        {
            "role": "user",
            "content": (
                "Here are the replies to summarize:\n\n"
                f"{bodies}\n\n"
                "Return only a JSON object with a single 'summary' field."
            ),
        },
    ]

def generate_json_response(messages, tokenizer, model, max_new_tokens=500, retries=3):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    for attempt in range(retries):
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        try:
            return json.loads(content)
        except JSONDecodeError:
            if attempt < retries - 1:
                print(f"⚠️ JSON parse failed on attempt {attempt + 1}. Retrying...")
            else:
                print("⚠️ All generation attempts failed.")
                return None

# ---------- paths ----------
input_path = Path("final_sets/RC_2025-12_politics_final_set.json")
output_path = Path("sim_output/RC_2025-12_politics_width_chain02.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

# ---------- load ----------
with open(input_path, "r", encoding="utf-8") as f:
    chains = json.load(f)

random.seed(42)

with open(output_path, "w", encoding="utf-8") as out_f:
    for i, chain in enumerate(chains):
        width_list = chain["ground_truth"]["width"]
        width_count = len(width_list)

        if width_count == 0:
            reply_summary = "No replies."
        else:
            message = makeMessage(width_list)
            reply_sum = generate_json_response(message, tokenizer, model)
            reply_summary = reply_sum["summary"] if reply_sum is not None else "Failed"

        chain["ground_truth"]["width"] = width_count
        chain["ground_truth"]["reply_summary"] = reply_summary

        out_f.write(json.dumps(chain, ensure_ascii=False) + "\n")
        out_f.flush()

        print(f"[{i+1}/{len(chains)}] Written to file.")