"""Train/evaluate one model over the four world-modeling tasks.

Two architectures, which is the comparison being asked for:

  --arch generative
      Every task is text. The model emits one JSON object carrying whichever of
      {score, controversiality, width, reply_summary} are enabled, and is
      trained with a plain causal-LM loss on prompt+target. Simple and uniform,
      but the numeric targets have to survive decoding and parsing.

  --arch hybrid
      Discriminative tasks get heads (two Huber regressions on log1p score and
      width, one class-weighted CE for controversiality) read off the pooled
      prompt representation; the generative task keeps the LM loss on the
      summary. One shared LoRA backbone, one joint loss. Numbers cannot fail to
      parse, which on the published data was an 81% failure mode.

`--tasks` selects the subset, so single-task runs (e.g. --tasks summary) use
exactly the same code path as the four-task runs and stay comparable.
"""
from __future__ import annotations
import argparse, datetime as dt, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("HF_HOME", "/scratch/daweili5/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from smwm.data.io import read_jsonl  # noqa: E402

ALL_TASKS = ["score", "controversiality", "width", "summary"]


def build_prompt(rec: dict, tasks: list[str]) -> str:
    stim = rec["stimulus"]; meta = rec.get("meta", {}); ctx = rec.get("context", [])[-4:]
    ts = int(meta.get("created_utc") or stim.get("created_utc") or 0)
    when = dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts else "unknown"
    lines = [f"- {(c.get('body') or '')[:400]}" for c in ctx] or ["- (none; top-level comment)"]
    want = []
    if "score" in tasks: want.append('  "score": <integer net upvotes>')
    if "controversiality" in tasks: want.append('  "controversiality": <0 or 1>')
    if "width" in tasks: want.append('  "width": <integer number of direct replies>')
    if "summary" in tasks: want.append('  "reply_summary": "<2-4 sentences describing the replies>"')
    return (
        "You are forecasting how a Reddit thread will unfold.\n\n"
        f"Subreddit: r/{meta.get('subreddit') or stim.get('subreddit')}\n"
        f"Posted: {when}\nDepth in thread: {meta.get('depth', len(rec.get('context', [])))}\n"
        f"Score of the parent comment: {(rec.get('context') or [{}])[-1].get('score', 'n/a')}\n"
        f"Score of the top-level comment: {meta.get('root_score', 'n/a')}\n\n"
        "Conversation so far (oldest first):\n" + "\n".join(lines) + "\n\n"
        f"The COMMENT just posted:\n{(stim.get('body') or '')[:1500]}\n\n"
        "Predict the engagement this comment will receive.\n"
        "Return ONLY a JSON object with exactly these fields:\n" + "\n".join(want)
    )


def build_target(rec: dict, tasks: list[str]) -> str:
    g = rec["ground_truth"]; o = {}
    if "score" in tasks: o["score"] = int(g["score"])
    if "controversiality" in tasks: o["controversiality"] = int(g["controversiality"])
    if "width" in tasks: o["width"] = int(g["width"])
    if "summary" in tasks:
        o["reply_summary"] = (rec.get("reply_summary") or g.get("reply_summary") or "")
    return json.dumps(o, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train"); ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--arch", default="generative", choices=["generative", "hybrid"])
    ap.add_argument("--tasks", default="score,controversiality,width,summary")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--zero-shot", action="store_true", help="skip training entirely")
    args = ap.parse_args()
    tasks = [t for t in args.tasks.split(",") if t in ALL_TASKS]

    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    if os.getenv("HUGGING_FACE"):
        from huggingface_hub import login; login(os.getenv("HUGGING_FACE"))

    ev = read_jsonl(args.eval)
    if args.eval_limit: ev = ev[: args.eval_limit]
    tr = []
    if args.train and not args.zero_shot:
        tr = read_jsonl(args.train)
        if args.train_limit: tr = tr[: args.train_limit]
    print(f"model={args.model} arch={args.arch} tasks={tasks} train={len(tr):,} eval={len(ev):,}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    hidden = model.config.hidden_size
    heads = None

    if tr:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules="q_proj k_proj v_proj o_proj".split()))
        model.print_trainable_parameters()
    model.cuda()

    num_tasks = [t for t in tasks if t in ("score", "width")]
    if args.arch == "hybrid":
        heads = nn.ModuleDict({
            **{t: nn.Linear(hidden, 1) for t in num_tasks},
            **({"controversiality": nn.Linear(hidden, 2)} if "controversiality" in tasks else {})
        }).float().cuda()

    # ------------------------------------------------------------------ train
    if tr:
        from torch.optim import AdamW
        gen_tasks = [t for t in tasks if t == "summary"] if args.arch == "hybrid" else tasks
        params = [p for p in model.parameters() if p.requires_grad]
        if heads is not None: params += list(heads.parameters())
        opt = AdamW(params, lr=args.lr)
        huber = nn.SmoothL1Loss()
        if args.arch == "hybrid" and "controversiality" in tasks:
            yc = np.array([int(r["ground_truth"]["controversiality"]) for r in tr])
            w = torch.tensor([1.0, max((yc == 0).sum(), 1) / max((yc == 1).sum(), 1)],
                             dtype=torch.float32, device="cuda")
            # summed then divided by batch size: reduction="mean" would divide by
            # the sum of weights and normalize the up-weighting straight back out
            ce_raw = nn.CrossEntropyLoss(weight=w, reduction="sum")
            ce = lambda lo, ta: ce_raw(lo, ta) / max(len(ta), 1)
        model.train()
        step = 0; t0 = time.time()
        for ep in range(args.epochs):
            order = np.random.permutation(len(tr))
            opt.zero_grad()
            for s in range(0, len(tr), args.batch_size):
                batch = [tr[i] for i in order[s : s + args.batch_size]]
                prompts = [build_prompt(r, tasks) for r in batch]
                loss = 0.0
                if gen_tasks:
                    txt = [p + "\n" + build_target(r, gen_tasks) + tok.eos_token
                           for p, r in zip(prompts, batch)]
                    enc = tok(txt, return_tensors="pt", padding=True, truncation=True,
                              max_length=args.max_length).to("cuda")
                    lab = enc.input_ids.clone(); lab[enc.attention_mask == 0] = -100
                    loss = loss + model(**enc, labels=lab).loss
                if args.arch == "hybrid" and heads is not None:
                    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                              max_length=args.max_length).to("cuda")
                    out = model(**enc, output_hidden_states=True)
                    h = out.hidden_states[-1]
                    m = enc.attention_mask.unsqueeze(-1).to(h.dtype)
                    pooled = ((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).float()
                    for t in num_tasks:
                        y = torch.tensor([math.log1p(max(float(r["ground_truth"][t]), 0.0))
                                          for r in batch], device="cuda")
                        loss = loss + huber(heads[t](pooled).squeeze(-1), y)
                    if "controversiality" in tasks:
                        y = torch.tensor([int(r["ground_truth"]["controversiality"]) for r in batch],
                                         device="cuda")
                        loss = loss + ce(heads["controversiality"](pooled), y)
                (loss / args.grad_accum).backward(); step += 1
                if step % args.grad_accum == 0:
                    opt.step(); opt.zero_grad()
                if step % 200 == 0:
                    print(f"[train] ep{ep} step {step} loss={float(loss):.4f} "
                          f"{(time.time()-t0)/60:.1f}m", flush=True)
            opt.step(); opt.zero_grad()
        print(f"[train] done in {(time.time()-t0)/60:.1f} min", flush=True)

    # ------------------------------------------------------------------- eval
    model.eval()
    if heads is not None: heads.eval()
    tok.padding_side = "left"
    gen_tasks = ["summary"] if args.arch == "hybrid" else tasks
    rows = []
    t0 = time.time()
    for s in range(0, len(ev), args.gen_batch):
        batch = ev[s : s + args.gen_batch]
        prompts = [build_prompt(r, tasks) for r in batch]
        pred = [{} for _ in batch]
        if gen_tasks:
            chat = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
                    for p in prompts]
            enc = tok(chat, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                   pad_token_id=tok.pad_token_id)
            for i, row in enumerate(g):
                raw = tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                pred[i]["_raw"] = raw
                try:
                    o = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
                    if isinstance(o, dict): pred[i].update(o)
                except Exception:
                    pass
        if args.arch == "hybrid" and heads is not None:
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length).to("cuda")
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
                h = out.hidden_states[-1]
                m = enc.attention_mask.unsqueeze(-1).to(h.dtype)
                pooled = ((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).float()
                for t in num_tasks:
                    v = np.expm1(heads[t](pooled).squeeze(-1).cpu().numpy())
                    for i in range(len(batch)): pred[i][t] = int(round(max(float(v[i]), 0)))
                if "controversiality" in tasks:
                    c = heads["controversiality"](pooled).argmax(-1).cpu().numpy()
                    for i in range(len(batch)): pred[i]["controversiality"] = int(c[i])
        for r, p in zip(batch, pred):
            rows.append({"record_id": r["record_id"], "predicted": p,
                         "ground_truth": r["ground_truth"],
                         "reference_summary": r.get("reply_summary", "")})
        if (s // max(args.gen_batch, 1)) % 10 == 0:
            d = s + len(batch); rate = d / max(time.time() - t0, 1e-9)
            print(f"[eval] {d}/{len(ev)} {rate:.2f} rec/s eta {(len(ev)-d)/max(rate,1e-9)/60:.0f}m",
                  flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({len(rows)} records)")


if __name__ == "__main__":
    main()
