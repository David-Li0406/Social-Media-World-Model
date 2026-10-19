"""LLM-as-a-judge over PREDICTED reply summaries, scored against reality.

Each record is rated twice so the two references can be compared:
  vs_replies  -- against the real reply texts (primary; reality, not model output)
  vs_summary  -- against a generated summary of those replies (the legacy target)

Reporting both matters: the legacy reference is itself Qwen output, so a gap
between the two columns measures how much the old protocol was rewarding
agreement with the summarizer rather than correct forecasting.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("HF_HOME", "/scratch/daweili5/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from smwm.data.io import read_jsonl  # noqa: E402
from smwm.eval.judge import build_judge_prompt, parse_rating  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="summary_subset.jsonl (replies + reference)")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--judge", default="Qwen/Qwen3-32B")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="HF engine only; vLLM serves bf16 and manages its own KV cache")
    ap.add_argument("--engine", default="auto", choices=["auto", "vllm", "hf"],
                    help="auto = use vLLM when importable, else fall back to transformers")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")

    # read_jsonl is gzip-aware; the subset ships as .jsonl.gz
    sub = {r["record_id"]: r for r in read_jsonl(args.subset)}
    preds = {r["record_id"]: r["predicted_summary"]
             for r in read_jsonl(args.predictions) if r.get("predicted_summary")}

    done = set()
    out_path = Path(args.output)
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            if l.strip():
                try: done.add(json.loads(l)["record_id"])
                except Exception: pass

    ids = [i for i in preds if i in sub and i not in done]
    if args.num_shards > 1:
        ids = [x for k, x in enumerate(sorted(ids)) if k % args.num_shards == args.shard]
    if args.limit:
        ids = ids[: args.limit]
    print(f"judging {len(ids):,} records with {args.judge}", flush=True)
    if not ids:
        return

    tok = AutoTokenizer.from_pretrained(args.judge)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    engine = args.engine
    if engine in ("auto", "vllm"):
        try:
            from vllm import LLM, SamplingParams
        except Exception as e:
            if engine == "vllm":
                sys.exit(f"--engine vllm requested but vLLM is unavailable: {e}")
            # Falling back keeps the pipeline runnable on a runner without vLLM
            # rather than failing the whole job over an optional accelerator.
            print(f"[judge] vLLM unavailable ({type(e).__name__}); falling back to transformers",
                  flush=True)
            engine = "hf"
        else:
            engine = "vllm"

    if engine == "vllm":
        print(f"[judge] engine=vLLM model={args.judge}", flush=True)
        llm = LLM(model=args.judge, dtype="bfloat16", trust_remote_code=True,
                  gpu_memory_utilization=args.gpu_mem_util, max_model_len=3072)
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

        def run(prompts):
            # vLLM does continuous batching internally, so hand it everything
            # at once instead of fixed-size chunks.
            outs = llm.generate([chat(p) for p in prompts], sp)
            return [o.outputs[0].text.strip() for o in outs]
    else:
        print(f"[judge] engine=transformers model={args.judge}", flush=True)
        kw = {"device_map": "cuda:0"}
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            kw["dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(args.judge, **kw); model.eval()

        def run(prompts):
            enc = tok([chat(p) for p in prompts], return_tensors="pt", padding=True,
                      truncation=True, max_length=3072).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     pad_token_id=tok.pad_token_id)
            return [tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                    for g in gen]

    # vLLM schedules its own batches; a large outer chunk just amortizes file IO.
    chunk_size = max(args.batch_size, 256) if engine == "vllm" else args.batch_size

    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for s in range(0, len(ids), chunk_size):
            chunk = ids[s : s + chunk_size]
            body = lambda i: (sub[i]["stimulus"].get("body") or "")
            ref = lambda i: (sub[i].get("reply_summary") or "")
            p_rep = [build_judge_prompt(body(i), ref(i), preds[i], replies=sub[i].get("replies"))
                     for i in chunk]
            p_sum = [build_judge_prompt(body(i), ref(i), preds[i]) for i in chunk]
            r_rep, r_sum = run(p_rep), run(p_sum)
            for i, a, b in zip(chunk, r_rep, r_sum):
                f.write(json.dumps({"record_id": i,
                                    "vs_replies": parse_rating(a),
                                    "vs_summary": parse_rating(b),
                                    "n_replies": len(sub[i].get("replies") or [])},
                                   ensure_ascii=False) + "\n")
            f.flush()
            d = s + len(chunk); rate = d / max(time.time() - t0, 1e-9)
            print(f"[judge] {d}/{len(ids)} {rate:.2f} rec/s eta {(len(ids)-d)/max(rate,1e-9)/60:.0f}m",
                  flush=True)

    rows = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    for col in ("vs_replies", "vs_summary"):
        v = [r[col] for r in rows if r.get(col)]
        if v:
            print(f"{col}: n={len(v)} mean={sum(v)/len(v):.3f} dist={dict(sorted(Counter(v).items()))}")


if __name__ == "__main__":
    main()
