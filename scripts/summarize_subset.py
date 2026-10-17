"""Phase B of the reply_summary target: generate the summaries. Needs a GPU.

Reuses the exact prompt from smwm.data.summarize_replies, so summaries here
match the definition of the legacy gold target and stay comparable to it.

Three differences from the original loop, which matter at 15k records:
  * resumable  -- appends, and skips record_ids already present, so a job that
                  dies after 9k records resumes instead of restarting
  * batched    -- generates batch_size prompts per forward pass (left-padded,
                  as decoder-only generation requires)
  * honest failures -- a record whose JSON never parses gets summary=null and
                  summary_ok=false, rather than the literal string "Failed"
                  sitting in the target field pretending to be a summary

bitsandbytes is unusable on this host (host GLIBC predates the shipped .so), so
the default is bf16; pass --load-in-4bit only where bnb actually imports.

Usage:
    PYTHONPATH=src python3 scripts/summarize_subset.py \
        --input /scratch/daweili5/smwm/data_multiyear_v2/summary_subset.jsonl \
        --output /scratch/daweili5/smwm/data_multiyear_v2/summary_subset_summarized.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from json.decoder import JSONDecodeError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("HF_HOME", "/scratch/daweili5/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from smwm.data.summarize_replies import _make_message  # noqa: E402

_FENCE_PREFIX = "```"


def parse_summary(raw: str):
    """Accept bare JSON, fenced JSON, or a bare string; else None."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith(_FENCE_PREFIX):
        text = text.strip("`")
        text = text[4:].lstrip() if text.lower().startswith("json") else text
    try:
        obj = json.loads(text)
    except (JSONDecodeError, ValueError):
        return None
    if isinstance(obj, dict):
        s = obj.get("summary")
        return s if isinstance(s, str) and s.strip() else None
    return obj if isinstance(obj, str) and obj.strip() else None


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.add(json.loads(line)["record_id"])
                except (JSONDecodeError, KeyError):
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-replies", type=int, default=20,
                    help="cap replies fed to the summarizer (long threads blow up the prompt)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0, help="this shard index (0-based)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the input N ways so one GPU can take each shard")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        sys.exit("No CUDA device. This step needs a GPU allocation "
                 "(or run it on the Machine B runner via the remote-gpu-exp workflow).")

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    out_path = Path(args.output)
    already = done_ids(out_path)
    skipped = [r for r in rows if r.get("summarizable") is False]
    todo = [r for r in rows if r["record_id"] not in already
            and r.get("summarizable") is not False]
    if args.num_shards > 1:
        todo = [r for i, r in enumerate(todo) if i % args.num_shards == args.shard]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows):,} records | {len(already):,} already summarized | "
          f"{len(skipped):,} unsummarizable (all replies removed) | {len(todo):,} to do")
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # required for batched decoder-only generation
    kw = {"device_map": "cuda:0"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()

    t0, n_ok, n_bad = time.time(), 0, 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fout:
        for s in range(0, len(todo), args.batch_size):
            batch = todo[s : s + args.batch_size]
            texts = [
                tok.apply_chat_template(
                    _make_message([x for x in r["replies"]
                                   if x["body"] not in ("[removed]", "[deleted]")
                                   ][: args.max_replies] or r["replies"][: args.max_replies]),
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for r in batch
            ]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=3072).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     pad_token_id=tok.pad_token_id)
            for r, row_ids in zip(batch, gen):
                raw = tok.decode(row_ids[enc.input_ids.shape[1]:],
                                 skip_special_tokens=True).strip()
                summary = parse_summary(raw)
                r["reply_summary"] = summary
                r["ground_truth"]["reply_summary"] = summary or ""
                r["summary_ok"] = summary is not None
                if summary is None:
                    r["summary_raw"] = raw[:400]
                    n_bad += 1
                else:
                    n_ok += 1
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            done = s + len(batch)
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(todo) - done) / max(rate, 1e-9) / 60
            print(f"[summarize] {done}/{len(todo)}  ok={n_ok} failed={n_bad}  "
                  f"{rate:.2f} rec/s  eta {eta:.0f} min", flush=True)

    print(f"\ndone: {n_ok} summarized, {n_bad} failed -> {out_path}")


if __name__ == "__main__":
    main()
