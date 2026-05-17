import math
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

MODEL_NAME = "Qwen/Qwen3-4B"   # replace with your exact model id
OUTPUT_DIR = "qwen_comment_sft"

# Load dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": "train.jsonl",
        "test": "test.jsonl",
    },
)

# Convert prompt/completion into single text sequence
def format_example(example):
    return {
        "text": example["prompt"] + "\n" + example["completion"]
    }

dataset = dataset.map(format_example)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    bf16=True,
    report_to="none",
    load_best_model_at_end=True,
    save_total_limit=2,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()

eval_results = trainer.evaluate()
print(eval_results)
if "eval_loss" in eval_results:
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")