"""Basic descriptive statistics and figures for the multi-year Reddit corpus.

Streams the per-cell JSONL so the full 443k-record corpus never sits in memory
at once. Reports size, target distributions, comment properties and coverage.

Outputs:
  data_analysis/dataset_stats.json
  slides/figures/ds_targets.png     score / reply-count / controversiality
  slides/figures/ds_coverage.png    records over time and by subreddit
  slides/figures/ds_comments.png    comment length and thread depth
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CORPUS = "/scratch/daweili5/smwm/data_multiyear_v2"


def stream(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect(corpus: Path):
    st = {
        "n": 0, "posts": set(), "authors": set(),
        "score": [], "width": [], "contr": 0, "nchars": [], "nwords": [], "gone": 0,
        "depth": Counter(),
        "by_month": defaultdict(lambda: {"n": 0, "contr": 0, "w0": 0}),
        "by_sub": defaultdict(lambda: {"n": 0, "contr": 0, "w0": 0, "score": []}),
        "cell": Counter(),
        "created": [],
    }
    files = [f for f in sorted(corpus.glob("*_train.jsonl")) + sorted(corpus.glob("*_test.jsonl"))
             if not f.name.startswith(("random_", "temporal_"))]
    for f in files:
        sub, month = f.name.replace("_train.jsonl", "").replace("_test.jsonl", "").rsplit("_", 1)
        for r in stream(f):
            g, m = r["ground_truth"], r["meta"]
            body = r["stimulus"].get("body") or ""
            st["n"] += 1
            st["posts"].add(m["link_id"])
            if r["stimulus"].get("author"):
                st["authors"].add(r["stimulus"]["author"])
            st["score"].append(g["score"]); st["width"].append(g["width"])
            st["contr"] += g["controversiality"]
            st["nchars"].append(len(body)); st["nwords"].append(len(body.split()))
            if body in ("[removed]", "[deleted]"):
                st["gone"] += 1
            st["depth"][min(len(r.get("context", [])), 6)] += 1
            st["created"].append(m["created_utc"])
            bm = st["by_month"][month]
            bm["n"] += 1; bm["contr"] += g["controversiality"]; bm["w0"] += (g["width"] == 0)
            bs = st["by_sub"][sub]
            bs["n"] += 1; bs["contr"] += g["controversiality"]; bs["w0"] += (g["width"] == 0)
            if len(bs["score"]) < 200000:
                bs["score"].append(g["score"])
            st["cell"][(month, sub)] += 1
    return st


def desc(a):
    a = np.asarray(a, float)
    return dict(mean=float(a.mean()), p50=float(np.median(a)),
                p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)),
                mn=float(a.min()), mx=float(a.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--figdir", default="slides/figures")
    args = ap.parse_args()

    st = collect(Path(args.corpus))
    score = np.asarray(st["score"], float)
    width = np.asarray(st["width"], float)
    nchars = np.asarray(st["nchars"], float)
    nwords = np.asarray(st["nwords"], float)
    months = sorted(st["by_month"])
    subs = sorted(st["by_sub"], key=lambda s: -st["by_sub"][s]["n"])

    import datetime as dt
    tmin = dt.datetime.fromtimestamp(min(st["created"]), dt.timezone.utc)
    tmax = dt.datetime.fromtimestamp(max(st["created"]), dt.timezone.utc)

    print("\n### Table 1 - Dataset size\n")
    print("| property | value |")
    print("|---|---:|")
    for k, v in [("records", f"{st['n']:,}"), ("posts (threads)", f"{len(st['posts']):,}"),
                 ("unique authors", f"{len(st['authors']):,}"),
                 ("subreddits", len(subs)), ("months", len(months)),
                 ("time span", f"{tmin:%Y-%m} to {tmax:%Y-%m}"),
                 ("train / test (random)", "354,658 / 88,553"),
                 ("train / test (temporal)", "383,083 / 60,000")]:
        print(f"| {k} | {v} |")

    print("\n### Table 2 - Prediction targets\n")
    print("| target | mean | median | p90 | p99 | max |")
    print("|---|---:|---:|---:|---:|---:|")
    s, w = desc(score), desc(width)
    print(f"| score (upvotes) | {s['mean']:.1f} | {s['p50']:.0f} | {s['p90']:.0f} | "
          f"{s['p99']:.0f} | {s['mx']:.0f} |")
    print(f"| width (direct replies) | {w['mean']:.2f} | {w['p50']:.0f} | {w['p90']:.0f} | "
          f"{w['p99']:.0f} | {w['mx']:.0f} |")
    print(f"| controversiality | {st['contr']/st['n']:.4f} | - | - | - | 1 |")
    print(f"\n- comments with no reply: {100*(width==0).mean():.1f}%")
    print(f"- controversial comments: {st['contr']:,} ({100*st['contr']/st['n']:.2f}%)")

    print("\n### Table 3 - Comment properties\n")
    print("| property | mean | median | p90 | max |")
    print("|---|---:|---:|---:|---:|")
    c, wd = desc(nchars), desc(nwords)
    print(f"| body length (characters) | {c['mean']:.0f} | {c['p50']:.0f} | {c['p90']:.0f} | {c['mx']:.0f} |")
    print(f"| body length (words) | {wd['mean']:.0f} | {wd['p50']:.0f} | {wd['p90']:.0f} | {wd['mx']:.0f} |")
    print(f"| removed / deleted body | {100*st['gone']/st['n']:.1f}% of records "
          f"({st['gone']:,}) | - | - | - |")
    print(f"| context depth (ancestors) | "
          f"{sum(k*v for k,v in st['depth'].items())/st['n']:.2f} | "
          f"{'0' if st['depth'][0] > st['n']/2 else '1'} | - | 6+ |")

    print("\n### Table 4 - By subreddit\n")
    print("| subreddit | records | share | median score | no reply | controversial |")
    print("|---|---:|---:|---:|---:|---:|")
    for sn in subs:
        b = st["by_sub"][sn]
        print(f"| r/{sn} | {b['n']:,} | {100*b['n']/st['n']:.1f}% | "
              f"{np.median(b['score']):.0f} | {100*b['w0']/b['n']:.1f}% | "
              f"{100*b['contr']/b['n']:.2f}% |")

    # ------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    fd = Path(args.figdir); fd.mkdir(parents=True, exist_ok=True)
    BLUE, ORANGE, GREEN = "#2E6F9E", "#E08A2E", "#4E9A6B"

    # Figure 1 - targets
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    pos = np.maximum(score, 0) + 1
    ax[0].hist(np.log10(pos), bins=60, color=BLUE)
    ax[0].set_xlabel("log10(score + 1)"); ax[0].set_ylabel("comments")
    ax[0].set_title("Score")
    bins = np.arange(0, 7)
    ax[1].bar(bins, [np.mean(np.clip(width, 0, 6) == b) for b in bins], color=ORANGE)
    ax[1].set_xticks(bins); ax[1].set_xticklabels([str(b) for b in bins[:-1]] + ["6+"])
    ax[1].set_xlabel("direct replies"); ax[1].set_ylabel("fraction of comments")
    ax[1].set_title("Reply count")
    rates = [100 * st["by_sub"][s]["contr"] / st["by_sub"][s]["n"] for s in subs]
    ax[2].barh([f"r/{s}" for s in subs][::-1], rates[::-1], color=GREEN)
    ax[2].set_xlabel("% controversial"); ax[2].set_title("Controversiality by subreddit")
    fig.tight_layout(); fig.savefig(fd / "ds_targets.png", dpi=150); plt.close(fig)

    # Figure 2 - coverage
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.0))
    bottom = np.zeros(len(months))
    for i, sn in enumerate(subs):
        vals = np.array([st["cell"].get((m, sn), 0) for m in months], float)
        ax[0].bar(range(len(months)), vals, bottom=bottom,
                  label=f"r/{sn}", color=plt.cm.tab20(i * 2))
        bottom += vals
    ax[0].set_xticks(range(len(months)))
    ax[0].set_xticklabels(months, rotation=90, fontsize=8)
    ax[0].set_ylabel("records"); ax[0].set_title("Records per month by subreddit")
    ax[0].set_ylim(0, max(bottom) * 1.35)
    ax[0].legend(fontsize=8, ncol=3, loc="upper center")
    counts = [st["by_sub"][s]["n"] for s in subs]
    ax[1].bar([f"r/{s}" for s in subs], counts, color=BLUE)
    ax[1].set_ylabel("records"); ax[1].set_title("Records per subreddit")
    ax[1].tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(fd / "ds_coverage.png", dpi=150); plt.close(fig)

    # Figure 3 - comment properties
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
    ax[0].hist(np.log10(np.maximum(nchars, 1)), bins=60, color=BLUE)
    ax[0].set_xlabel("log10(body length in characters)"); ax[0].set_ylabel("comments")
    ax[0].set_title("Comment length")
    ax[0].axvline(np.log10(9), color="#B03A2E", ls="--", lw=1.4)
    ax[0].annotate(f"[removed] / [deleted]\n{100*st['gone']/st['n']:.0f}% of records",
                   xy=(np.log10(9), ax[0].get_ylim()[1] * 0.80),
                   xytext=(1.55, ax[0].get_ylim()[1] * 0.80), fontsize=9, color="#B03A2E",
                   arrowprops=dict(arrowstyle="->", color="#B03A2E"))
    dk = sorted(st["depth"])
    ax[1].bar(dk, [st["depth"][k] / st["n"] for k in dk], color=ORANGE)
    ax[1].set_xticks(dk); ax[1].set_xticklabels([str(k) for k in dk[:-1]] + ["6+"])
    ax[1].set_xlabel("ancestor comments above the stimulus")
    ax[1].set_ylabel("fraction of comments"); ax[1].set_title("Thread depth")
    fig.tight_layout(); fig.savefig(fd / "ds_comments.png", dpi=150); plt.close(fig)

    payload = {
        "n_records": st["n"], "n_posts": len(st["posts"]), "n_authors": len(st["authors"]),
        "months": months, "subreddits": subs, "span": [str(tmin.date()), str(tmax.date())],
        "score": desc(score), "width": desc(width),
        "body_chars": desc(nchars), "body_words": desc(nwords),
        "removed_or_deleted": st["gone"], "removed_or_deleted_frac": st["gone"] / st["n"],
        "width_zero_frac": float((width == 0).mean()),
        "contr_rate": st["contr"] / st["n"], "contr_positives": st["contr"],
        "depth_dist": {str(k): v for k, v in sorted(st["depth"].items())},
        "by_month": {m: dict(st["by_month"][m]) for m in months},
        "by_subreddit": {s: {k: v for k, v in st["by_sub"][s].items() if k != "score"}
                         for s in subs},
    }
    Path("data_analysis").mkdir(exist_ok=True)
    Path("data_analysis/dataset_stats.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote data_analysis/dataset_stats.json and 3 figures to {fd}/")


if __name__ == "__main__":
    main()
