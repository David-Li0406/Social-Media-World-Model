"""Predict what replies a comment WILL receive, from the comment + metadata only.

This is the forecasting form of the reply_summary target: the model never sees
the replies. It sees the ancestor chain, the stimulus comment, and the metadata
available at posting time (subreddit, timestamp, depth, parent/root score) and
must describe the replies the comment is going to draw.

Deliberately excluded from the prompt: the stimulus's own score and
controversiality (both are prediction targets) and the replies themselves.

Output is judged against the real replies and against a reference summary of
them -- see scripts/judge_summaries.py --use-replies.
"""
from __future__ import annotations

import argparse, datetime as dt, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("HF_HOME", "/scratch/daweili5/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_prompt(rec: dict, max_ctx: int = 4) -> str:
    stim = rec["stimulus"]; meta = rec.get("meta", {})
    ctx = rec.get("context", [])[-max_ctx:]
    ts = int(meta.get("created_utc") or stim.get("created_utc") or 0)
    when = dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts else "unknown"
    lines = [f"- {(c.get('body') or '')[:400]}" for c in ctx] or ["- (none; this is a top-level comment)"]
    return (
        "You are forecasting how a Reddit thread will unfold.\n\n"
        f"Subreddit: r/{meta.get('subreddit') or stim.get('subreddit')}\n"
        f"Posted: {when}\n"
        f"Depth in thread: {meta.get('depth', len(rec.get('context', [])))}\n"
        f"Score of the parent comment: {(rec.get('context') or [{}])[-1].get('score', 'n/a')}\n"
        f"Score of the top-level comment in this thread: {meta.get('root_score', 'n/a')}\n\n"
        "Conversation so far (oldest first):\n" + "\n".join(lines) + "\n\n"
        "The COMMENT that was just posted:\n"
        f"{(stim.get('body') or '')[:1500]}\n\n"
        "Predict the replies this comment will receive. Describe the key points "
        "people will make and the overall sentiment toward the comment. Do not "
        "restate the comment itself.\n"
        'Return ONLY a JSON object: {"summary": "<2-4 sentences>"}'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from summarize_subset import done_ids, parse_summary

    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    out_path = Path(args.output); already = done_ids(out_path)
    todo = [r for r in rows if r["record_id"] not in already]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows):,} records | {len(already):,} done | {len(todo):,} to predict", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    t0, ok, bad = time.time(), 0, 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), args.batch_size):
            batch = todo[s : s + args.batch_size]
            texts = [tok.apply_chat_template([{"role": "user", "content": build_prompt(r)}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=3072).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     pad_token_id=tok.pad_token_id)
            for r, row in zip(batch, gen):
                raw = tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                pred = parse_summary(raw) or raw[:600] or None
                ok += pred is not None; bad += pred is None
                f.write(json.dumps({"record_id": r["record_id"],
                                    "predicted_summary": pred,
                                    "model": args.model}, ensure_ascii=False) + "\n")
            f.flush()
            d = s + len(batch); rate = d / max(time.time() - t0, 1e-9)
            print(f"[predict] {d}/{len(todo)} {rate:.2f} rec/s eta {(len(todo)-d)/max(rate,1e-9)/60:.0f}m",
                  flush=True)
    print(f"done: {ok} predicted, {bad} unparsed -> {out_path}")


if __name__ == "__main__":
    main()
