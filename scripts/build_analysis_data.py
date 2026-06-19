"""Build cross-domain + temporal test sets from the Reddit dump archive.

All test-only (test_fraction=1.0), numeric targets, capped at max_chains.
  - Cross-domain (RC_2025-09): politics, news, worldnews, Conservative, technology
  - Temporal (politics): RC_2025-09, RC_2025-02, RC_2024-08
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from smwm.data import build_domain  # noqa: E402

ARCHIVE = "/scratch/daweili5/parquet_v2"
OUT = "data_analysis/domains"
MAX_CHAINS = 2000

JOBS = [
    # (parquet_month, subreddit, tag)
    ("RC_2025-09", "politics", "politics_2025-09"),
    ("RC_2025-09", "news", "news_2025-09"),
    ("RC_2025-09", "worldnews", "worldnews_2025-09"),
    ("RC_2025-09", "Conservative", "Conservative_2025-09"),
    ("RC_2025-02", "politics", "politics_2025-02"),
    ("RC_2024-08", "politics", "politics_2024-08"),
]

for month, sub, tag in JOBS:
    out_test = Path(OUT) / f"{tag}_test.jsonl"
    if out_test.exists():
        print(f"[skip] {tag} exists", flush=True)
        continue
    try:
        r = build_domain.run(f"{ARCHIVE}/{month}.parquet", sub, OUT, tag,
                             test_fraction=1.0, max_chains=MAX_CHAINS)
        print(f"[done] {r}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {tag}: {e!r}", flush=True)
print("ALL_BUILDS_DONE", flush=True)
