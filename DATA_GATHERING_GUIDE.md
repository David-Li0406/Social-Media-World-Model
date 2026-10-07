# Reddit Data Gathering & Extraction Guide

**Project**: ASU Cyberbullying Detection Research
**Author**: <your name>
**Purpose**: End-to-end reproducible guide for downloading Reddit comment dumps and converting them to queryable Parquet files. Written so that a Claude Code agent (or any developer) can pick this up and reproduce the entire pipeline from scratch.

---

## Overview

Reddit's full comment history is available as monthly `.zst` (Zstandard-compressed) NDJSON files via the Academic Torrents / Pushshift archive. Each line in a `.zst` file is a single JSON object representing one Reddit comment. This pipeline downloads those dumps and converts them into clean, fixed-schema Parquet files suitable for DuckDB queries, pandas analysis, or any columnar data tool.

**Pipeline summary**:

```
Torrent download (.zst files, ~3.5 TB)
    → convert_all.py (streaming decompress → JSON parse → schema enforcement → Parquet)
    → Clean .parquet files (one per .zst, ~15 fields, ZSTD-compressed)
    → DuckDB / pandas queries for subreddit-specific analysis
```

---

## Phase 1: Data Download

### Source

The Reddit comment dumps come from the Pushshift archive, distributed via Academic Torrents. Each monthly file is named like:

- `RC_2005-06.zst` (comments from June 2005)
- `RC_2025-07.zst` (comments from July 2025)

"RC" = Reddit Comments. There are also "RS" files (Reddit Submissions / posts) — this pipeline handles RC files only. RS files have a different schema (`selftext`, `title`, `url` instead of `body`).

### Download method: Torrent

Use a torrent client that supports large multi-file torrents. On an HPC cluster like ASU SOL, use a CLI torrent client.

**Option A: aria2 (recommended for HPC)**
```bash
# Install aria2 if not available
conda install -c conda-forge aria2

# Download using the torrent magnet link or .torrent file
aria2c --seed-time=0 --dir=/scratch/rkarumur/data \
       --max-concurrent-downloads=4 \
       --split=4 \
       "path/to/reddit_comments.torrent"
```

**Option B: transmission-cli**
```bash
transmission-cli -w /scratch/rkarumur/data "path/to/reddit_comments.torrent"
```

**Option C: rtorrent (background daemon)**
```bash
rtorrent "path/to/reddit_comments.torrent"
# Configure download directory in .rtorrent.rc
```

### Torrent source URLs

Search Academic Torrents for "Pushshift Reddit" or "Reddit comments" to find the latest available torrent. The archive is organized by year/month. As of this pipeline's execution, files ranged from `RC_2005-06.zst` through `RC_2025-07.zst` — 488 files totaling approximately 3.5 TB compressed.

### Important: Wait for downloads to complete

Do NOT start the conversion pipeline on files that are still downloading. Partially downloaded `.zst` files will produce zstd decompression errors like:

```
ERROR: Unknown frame descriptor
ERROR: Data corruption detected
```

Verify a file is complete before converting:
```bash
# Check if file size matches the expected size from the torrent
ls -lh /scratch/rkarumur/data/RC_2025-12.zst

# Quick integrity check — try decompressing the first few lines
zstd -d --stdout RC_2025-12.zst | head -1 | python3 -c "import sys,json; json.loads(sys.stdin.readline()); print('OK')"
```

### Storage requirements

| What | Size |
|------|------|
| Raw `.zst` files | ~3.5 TB |
| Output `.parquet` files | ~500 GB–1 TB (varies by compression ratio) |
| Total scratch space needed | ~5 TB minimum |

On ASU SOL, use `/scratch/rkarumur/` which has a 100 TB quota. Do NOT use `$HOME` (50 GB quota).

---

## Phase 2: Environment Setup

### Python environment

```bash
# Create a conda/mamba environment
mamba create -n reddit python=3.11 -y
mamba activate reddit

# Install dependencies
pip install pyarrow zstandard
```

**Required packages**:

| Package | Purpose | Min version |
|---------|---------|-------------|
| `pyarrow` | Parquet reading/writing, Arrow table construction | 14.0+ |
| `zstandard` | Streaming `.zst` decompression | 0.22+ |

Standard library modules used: `io`, `json`, `os`, `time`, `multiprocessing`.

### HPC setup (ASU SOL specific)

Create a SLURM batch script `convert_all.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=reddit-parquet
#SBATCH --partition=highmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=convert_%j.out
#SBATCH --error=convert_%j.err

module load mamba
mamba activate reddit

python convert_all.py
```

Submit with:
```bash
sbatch convert_all.slurm
```

**Why highmem?** Some monthly files (especially 2023–2025) are extremely large and the JSON parsing + Arrow table construction can spike memory. 128 GB gives headroom. Earlier runs with less memory were OOM-killed at ~67 GB RSS.

**Why 48 hours?** Converting all 488 files in one run can take 20–40 hours depending on I/O speed. The pipeline is restart-safe, so if the job times out, just resubmit — it picks up where it left off.

---

## Phase 3: The Conversion Script — `convert_all.py`

### What it does

Converts each `.zst` file into a `.parquet` file with a fixed 15-field schema. One input file → one output file, no merging.

### Full script

```python
#!/usr/bin/env python3
"""
Reddit .zst -> Parquet converter

Fixed 15-field schema for cyberbullying research:
  14 native Reddit fields + _source_file for lineage/debugging.

- 1:1 mapping: each .zst produces exactly one .parquet file
- Fixed schema eliminates OOM (no nested structs) and type mismatch issues
- Parallel processing via multiprocessing
- Tracks completed files in .done.txt — restart-safe
- Designed for ASU SOL /scratch filesystem

Usage:
  sbatch convert_all.slurm
"""

import io
import json
import os
import time
from multiprocessing import Pool, cpu_count

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

# ========================
# CONFIG
# ========================
DATA_DIR   = "/scratch/rkarumur/data"        # ← CHANGE: where your .zst files live
OUTPUT_DIR = "/scratch/rkarumur/parquet_v2"   # ← CHANGE: where .parquet files go
DONE_FILE  = os.path.join(OUTPUT_DIR, ".done.txt")

NUM_WORKERS = min(8, cpu_count())
FLUSH_EVERY = 100_000   # safe now — each row is tiny with fixed schema

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# FIXED SCHEMA — all fields from Reddit's raw JSON, no fabrication
# ========================
SCHEMA = pa.schema([
    ("id",                pa.string()),    # comment's unique ID
    ("parent_id",         pa.string()),    # t1_ = reply to comment, t3_ = reply to post
    ("link_id",           pa.string()),    # the post (submission) this lives under
    ("subreddit",         pa.string()),    # community name
    ("author",            pa.string()),    # username
    ("body",              pa.string()),    # comment text
    ("score",             pa.int64()),     # net votes (fuzzed by Reddit)
    ("controversiality",  pa.int64()),     # 1 = controversial (roughly equal up/down)
    ("created_utc",       pa.int64()),     # unix timestamp
    ("distinguished",     pa.string()),    # "moderator", "admin", or null
    ("edited",            pa.string()),    # "false" or unix timestamp as string
    ("author_flair_text", pa.string()),    # community flair
    ("gilded",            pa.int64()),     # number of times gilded
    ("stickied",          pa.bool_()),     # pinned by mod
    ("_source_file",      pa.string()),    # which .zst file (for debugging)
])

FIELDS = [f.name for f in SCHEMA]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ========================
# ROW EXTRACTION
# ========================
def extract_row(obj, source_file):
    """
    Pull exactly the 15 fields from a raw Reddit JSON object.
    All values are cast to their target types to avoid Arrow errors.
    Fields missing from older dumps become None.
    """
    edited_raw = obj.get("edited")
    if edited_raw is False or edited_raw is None:
        edited = "false"
    else:
        edited = str(edited_raw)

    def to_int(val):
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def to_bool(val):
        if val is None:
            return None
        if isinstance(val, bool):
            return val
        return bool(val)

    return (
        str(obj.get("id", "")) or None,
        str(obj.get("parent_id", "")) or None,
        str(obj.get("link_id", "")) or None,
        str(obj.get("subreddit", "")) or None,
        str(obj.get("author", "")) or None,
        obj.get("body"),
        to_int(obj.get("score")),
        to_int(obj.get("controversiality")),
        to_int(obj.get("created_utc")),
        obj.get("distinguished"),
        edited,
        obj.get("author_flair_text"),
        to_int(obj.get("gilded")),
        to_bool(obj.get("stickied")),
        source_file,
    )


# ========================
# DONE TRACKING
# ========================
def load_done_set():
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def mark_done(path):
    with open(DONE_FILE, "a") as f:
        f.write(path + "\n")


# ========================
# SINGLE FILE CONVERTER
# ========================
def convert_one(zst_path):
    """Convert one .zst to one .parquet with fixed schema."""
    name = os.path.basename(zst_path)
    out_name = name.replace(".zst", ".parquet")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    tmp_path = out_path + ".tmp"

    total = 0
    errors = 0
    writer = None

    n_cols = len(FIELDS)
    buffers = [[] for _ in range(n_cols)]

    try:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)

        with open(zst_path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(
                    reader, encoding="utf-8", errors="replace"
                )

                for line in text_stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        row = extract_row(obj, name)
                        for i, val in enumerate(row):
                            buffers[i].append(val)
                        total += 1
                    except (json.JSONDecodeError, Exception):
                        errors += 1
                        continue

                    if total % FLUSH_EVERY == 0:
                        writer = _flush(buffers, writer, tmp_path)
                        buffers = [[] for _ in range(n_cols)]

                if buffers[0]:
                    writer = _flush(buffers, writer, tmp_path)

        if writer is not None:
            writer.close()
            writer = None

        os.rename(tmp_path, out_path)
        return (zst_path, total, errors)

    except Exception as e:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return (zst_path, f"ERROR: {e}", 0)


def _flush(buffers, writer, tmp_path):
    """Write buffered columns to parquet."""
    arrays = []
    for i, field in enumerate(SCHEMA):
        arrays.append(pa.array(buffers[i], type=field.type))

    table = pa.table(
        {field.name: arrays[i] for i, field in enumerate(SCHEMA)},
        schema=SCHEMA,
    )

    if writer is None:
        writer = pq.ParquetWriter(tmp_path, SCHEMA, compression="zstd")
    writer.write_table(table)
    return writer


# ========================
# MAIN
# ========================
def main():
    log(f"Scanning {DATA_DIR} for .zst files...")
    all_zst = []
    for dirpath, _, filenames in os.walk(DATA_DIR):
        for fn in filenames:
            if fn.endswith(".zst"):
                all_zst.append(os.path.join(dirpath, fn))

    all_zst.sort()
    log(f"Found {len(all_zst)} .zst files total")

    done = load_done_set()
    todo = [f for f in all_zst if f not in done]
    log(f"Already done: {len(done)}, remaining: {len(todo)}")

    if not todo:
        log("Nothing to do — all files already converted.")
        return

    todo.sort(key=lambda p: os.path.getsize(p))

    log(f"Starting {NUM_WORKERS} workers...")
    log(f"Schema: {', '.join(FIELDS)}")
    t0 = time.time()
    completed = 0
    failed = 0

    with Pool(processes=NUM_WORKERS) as pool:
        for result in pool.imap_unordered(convert_one, todo):
            path = result[0]
            name = os.path.basename(path)

            if isinstance(result[1], int):
                rows, errs = result[1], result[2]
                mark_done(path)
                completed += 1
                elapsed = time.time() - t0
                rate = completed / elapsed * 3600
                log(f"[{completed}/{len(todo)}] OK  {name}: {rows:>12,} rows  "
                    f"({errs} parse errors)  [{rate:.0f} files/hr]")
            else:
                failed += 1
                log(f"[{completed}/{len(todo)}] FAIL {name}: {result[1]}")

    elapsed = time.time() - t0
    log(f"Done in {elapsed/3600:.1f}h — {completed} converted, {failed} failed")


if __name__ == "__main__":
    main()
```

### Configuration — what to change

| Variable | What to set it to |
|----------|-------------------|
| `DATA_DIR` | Directory containing downloaded `.zst` files |
| `OUTPUT_DIR` | Directory where `.parquet` files will be written |
| `NUM_WORKERS` | Number of parallel processes (default: 8 or CPU count, whichever is lower) |
| `FLUSH_EVERY` | Rows to buffer before writing to Parquet (default: 100,000) |

---

## Phase 4: Running the Pipeline

### Step 1: Verify downloads are complete

```bash
# Count .zst files
find /scratch/rkarumur/data -name "*.zst" | wc -l

# Check for suspiciously small files (may be incomplete)
find /scratch/rkarumur/data -name "*.zst" -size -1M -ls
```

### Step 2: Run the conversion

```bash
# Interactive (small batches / testing)
python convert_all.py

# Via SLURM (production)
sbatch convert_all.slurm
```

### Step 3: Monitor progress

```bash
# Watch the output log
tail -f convert_*.out

# Count completed files
wc -l /scratch/rkarumur/parquet_v2/.done.txt

# Count parquet files
ls /scratch/rkarumur/parquet_v2/*.parquet | wc -l
```

### Step 4: Handle failures and restart

The pipeline is restart-safe via `.done.txt`. If a job times out or gets OOM-killed:

```bash
# Just resubmit — it picks up where it left off
sbatch convert_all.slurm
```

If `.done.txt` has stale entries from genuinely failed files (files that errored but got logged):

```bash
# Remove entries for files that don't have a corresponding .parquet
cd /scratch/rkarumur/parquet_v2
awk -F/ '{f=$NF; sub(/\.zst$/,".parquet",f); if (system("test -f " f) != 0) next} 1' .done.txt > .done_clean.txt
mv .done_clean.txt .done.txt
```

---

## Phase 5: Querying the Output

### DuckDB (recommended for large-scale queries)

```python
import duckdb

con = duckdb.connect()
con.execute("PRAGMA memory_limit='3GB'")
con.execute("PRAGMA threads=2")

# Query a single file
con.execute("""
    SELECT subreddit, COUNT(*) as n
    FROM read_parquet('/scratch/rkarumur/parquet_v2/RC_2025-12.parquet')
    GROUP BY subreddit
    ORDER BY n DESC
    LIMIT 20
""").fetchdf()

# Query all files at once (glob pattern)
con.execute("""
    SELECT subreddit, COUNT(*) as n
    FROM read_parquet('/scratch/rkarumur/parquet_v2/*.parquet')
    GROUP BY subreddit
    ORDER BY n DESC
    LIMIT 20
""").fetchdf()

# Extract a subreddit subset
con.execute("""
    COPY (
        SELECT * FROM read_parquet('/scratch/rkarumur/parquet_v2/RC_2025-12.parquet')
        WHERE subreddit = 'AmItheAsshole'
    ) TO 'aita.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
```

### Pandas (smaller subsets)

```python
import pyarrow.parquet as pq

# Read specific columns only (memory-efficient)
table = pq.read_table(
    'RC_2025-12.parquet',
    columns=['author', 'body', 'score', 'subreddit']
)
df = table.to_pandas()
```

---

## Schema Reference

### The 15 fields

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `id` | string | Comment's unique ID (without prefix) |
| 2 | `parent_id` | string | `t1_xxx` = reply to comment `xxx`, `t3_xxx` = reply to post `xxx` |
| 3 | `link_id` | string | The post (submission) this comment belongs to |
| 4 | `subreddit` | string | Community name (without `r/` prefix) |
| 5 | `author` | string | Username (without `u/` prefix) |
| 6 | `body` | string | Full comment text (Markdown) |
| 7 | `score` | int64 | Net votes (upvotes minus downvotes, fuzzed by Reddit) |
| 8 | `controversiality` | int64 | `1` if roughly equal up/down votes, `0` otherwise |
| 9 | `created_utc` | int64 | Unix timestamp of comment creation |
| 10 | `distinguished` | string | `"moderator"`, `"admin"`, or `null` |
| 11 | `edited` | string | `"false"` if never edited, unix timestamp as string if edited |
| 12 | `author_flair_text` | string | Community-specific flair text |
| 13 | `gilded` | int64 | Number of times the comment was gilded (awarded) |
| 14 | `stickied` | bool | `true` if pinned by a moderator |
| 15 | `_source_file` | string | Which `.zst` file this row came from (for lineage) |

### Why this schema?

Reddit's raw JSON contains 50+ fields, many of which are nested objects (e.g., `gildings`, `author_flair_richtext`). Naively converting the full JSON to Parquet causes:

- **OOM errors**: Nested structs with unpredictable depth blow up memory
- **Schema mismatch errors**: Different months have different fields; Arrow can't reconcile them
- **Type conflicts**: The `edited` field is `false` (bool) when not edited but a Unix timestamp (int/float) when edited — Arrow chokes on mixed types

The fixed 15-field schema solves all of these by extracting only the fields needed for cyberbullying research, casting everything to predictable types, and dropping nested objects entirely.

---

## Reconstructing Reddit Thread Trees

Once you have the Parquet files, you can reconstruct the thread structure (who replied to whom) using the `parent_id` prefix system:

```python
import duckdb

con = duckdb.connect()

# Get all comments in a thread
thread = con.execute("""
    SELECT id, parent_id, link_id, author, body, score, controversiality
    FROM read_parquet('RC_2025-12.parquet')
    WHERE link_id = 't3_1phuxat'
    ORDER BY score DESC
""").fetchdf()

# Build children map
children = {}
for _, row in thread.iterrows():
    pid = row['parent_id']
    # Strip prefix: t1_xxx → xxx (reply to comment), t3_xxx → xxx (reply to post)
    parent_key = pid[3:] if pid else None
    children.setdefault(parent_key, []).append(row)

# The root is the post ID itself (link_id without prefix)
root_id = '1phuxat'  # from t3_1phuxat
root_children = children.get(root_id, [])
```

### ID prefix system

| Prefix | Meaning |
|--------|---------|
| `t1_` | Comment (reply to another comment) |
| `t3_` | Post/submission (reply to the original post) |
| `t2_` | Account (not used in comment data) |
| `t4_` | Message (not used in comment data) |
| `t5_` | Subreddit (not used in comment data) |

---

## Known Issues & Solutions

### Issue 1: Zstd decompression errors on partially downloaded files

```
ERROR: Unknown frame descriptor
ERROR: Data corruption detected
```

**Cause**: The `.zst` file is still being downloaded by the torrent client.
**Fix**: Wait for the download to complete, or exclude in-progress files.

### Issue 2: `edited` field type conflict

```
Could not convert 1525176769 with type int: tried to convert to boolean
```

**Cause**: Reddit's `edited` field is `False` (Python bool) when not edited, but a Unix timestamp (int) when edited. Arrow can't handle mixed types in one column.
**Fix**: The `extract_row()` function casts `edited` to a string: `"false"` or the timestamp as a string.

### Issue 3: Empty struct `gildings` field

```
Cannot write struct type 'gildings' with no child field to Parquet
```

**Cause**: Newer Reddit dumps include a `gildings` field that's sometimes an empty struct `{}`.
**Fix**: The fixed schema drops `gildings` entirely (we keep `gilded` which is the integer count).

### Issue 4: OOM kills on SLURM

```
slurmstepd: error: Detected 1 oom_kill event in StepId=...
```

**Cause**: Large monthly files (especially 2023–2025, 100M+ comments each) can spike memory during Arrow table construction.
**Fix**: Use `highmem` partition with 128 GB RAM. The 100K row flush interval keeps memory bounded per worker; 128 GB gives enough headroom for 8 parallel workers.

### Issue 5: Stale `.done.txt` entries

If a file was partially converted (errored mid-way) but still got an entry in `.done.txt` from a previous run, the pipeline will skip it even though the `.parquet` file doesn't exist or is corrupt.

**Fix**: Clean the ledger:
```bash
# Keep only entries whose .parquet file actually exists
cd /scratch/rkarumur/parquet_v2
for f in $(cat .done.txt); do
    pq=$(basename "$f" .zst).parquet
    [ -f "$pq" ] && echo "$f"
done > .done_clean.txt
mv .done_clean.txt .done.txt
```

---

## Verification

After the pipeline completes, verify the output:

```bash
# Count: should match number of .zst files (488 in original run)
ls /scratch/rkarumur/parquet_v2/*.parquet | wc -l

# Spot-check a file
python3 -c "
import pyarrow.parquet as pq
t = pq.read_metadata('parquet_v2/RC_2025-12.parquet')
print(f'Rows: {t.num_rows:,}')
print(f'Columns: {t.num_columns}')
print(f'Schema: {t.schema.to_arrow_schema()}')
"

# Query with DuckDB
python3 -c "
import duckdb
r = duckdb.sql(\"SELECT COUNT(*) as n FROM read_parquet('parquet_v2/RC_2025-12.parquet')\").fetchone()
print(f'RC_2025-12: {r[0]:,} comments')
"
```

---

## Appendix: Actual Commands Run on ASU SOL

These are the exact commands from the original pipeline execution, extracted from the SOL terminal history and annotated by phase. The torrent file used was `3d426c47c767d40f82c7ef0f47c3acacedd2bf44.torrent`.

### A. HPC Session & Environment Setup

```bash
# SSH into SOL
ssh <your_asurite>@login.sol.rc.asu.edu

# Get an interactive session on highmem partition
srun --partition=highmem \
     --nodes=1 --ntasks=1 --cpus-per-task=32 \
     --mem=32G --time=24:00:00 --pty bash

# Load and create conda environment
module load mamba/latest
mamba create -n reddit-pipeline python=3.10 -y
source activate reddit-pipeline

# Install dependencies
pip install pyarrow zstandard pandas
```

### B. Torrent Download

```bash
# Load aria2 for torrent downloading
module load aria2-1.36.0-gcc-12.1.0

# Inspect torrent contents — count .zst files available
aria2c -S /scratch/rkarumur/3d426c47c767d40f82c7ef0f47c3acacedd2bf44.torrent | grep '.zst' | wc -l

# Download via aria2 (ran inside tmux for persistence)
tmux new -s reddit
aria2c --dir=/scratch/rkarumur/data \
       /scratch/rkarumur/3d426c47c767d40f82c7ef0f47c3acacedd2bf44.torrent

# Monitor download progress
du -sh /scratch/rkarumur/data/*
du -sh /scratch/rkarumur/data/reddit/*

# Verify file counts
ls /scratch/rkarumur/data/reddit/comments | wc -l    # RC files (comments)
ls /scratch/rkarumur/data/reddit/submissions | wc -l  # RS files (posts)

# Total .zst files downloaded
find /scratch/rkarumur/data -name '*.zst' | wc -l
```

### C. Early Pipeline Attempts (pipeline5.py — abandoned)

```bash
# First attempt: full-schema conversion (caused OOM + schema mismatch errors)
vi pipeline5.py
nohup python /scratch/rkarumur/pipeline5.py > pipeline_log 2>&1 &

# Monitor
tail -n 100 pipeline_log
watch -n 1 "tail -n 100 pipeline_log"
watch -n 2 "ls parquet/ | wc -l"

# Check progress
ls -lhS parquet/
ls parquet/ | wc -l
du -sh /scratch/rkarumur/parquet/

# This approach failed — OOM kills, schema mismatches, nested struct errors
# Abandoned in favor of convert_all.py with fixed 15-field schema
kill -9 1251611
rm -rf data/ parquet/  # cleaned up for restart with new approach
```

### D. Convert Pipeline v2 (convert_all.py — production)

```bash
# Write the fixed-schema conversion script
vi convert_all.py

# Create SLURM batch script for convert_all.py
cat << 'EOF' > /scratch/rkarumur/convert_all.slurm
#!/bin/bash
#SBATCH --job-name=convert_v2
#SBATCH --partition=highmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=convert_v2_%j.out
#SBATCH --error=convert_v2_%j.err

module load mamba/latest
source activate reddit-pipeline
python /scratch/rkarumur/convert_all.py
EOF

# Submit job
sbatch convert_all.slurm
squeue -u $USER   # replace with your ASURITE

# Monitor conversion
cd /scratch/rkarumur/parquet_v2/
watch -n 10 "ls | wc -l"
wc -l /scratch/rkarumur/parquet_v2/.done.txt
tail -f /scratch/rkarumur/convert_v2_<JOBID>.out

# Check progress: done count vs parquet file count
wc -l /scratch/rkarumur/parquet_v2/.done.txt && \
ls /scratch/rkarumur/parquet_v2/*.parquet 2>/dev/null | wc -l

# Inspect output sizes
du -sh parquet_v2/* | head -n 20

# Check for stuck .tmp files (incomplete conversions)
ls -lhS /scratch/rkarumur/parquet_v2/*.tmp 2>/dev/null

# Count successes and failures in log
grep "OK " pipeline_log_<JOBID>.out | wc -l
grep "FAIL" pipeline_log_<JOBID>.out | wc -l
```

### E. Handling Failures & Restarts

```bash
# Multiple SLURM jobs were needed (OOM kills, timeouts)
# Each resubmission picks up where it left off via .done.txt

# Clean stale .done.txt entries (files marked done but .parquet missing)
cd /scratch/rkarumur/parquet_v2/
cp .done.txt .done.txt.bak
awk -F/ '{f=$NF; sub(/\.zst$/,".parquet",f); if (system("test -f " f) == 0) print}' \
    .done.txt.bak > .done.txt
diff .done.txt .done.txt.bak

# Nuclear restart when needed (early failed runs)
rm -rf /scratch/rkarumur/parquet_v2/*

# Resubmit
sbatch convert_all.slurm
```

### F. Post-Conversion Verification & Permissions

```bash
# Final count
ls /scratch/rkarumur/parquet_v2/ | wc -l   # → 488

# Set permissions for collaborators
chmod o+rx /scratch/$USER
chmod -R o+rx /scratch/rkarumur/parquet_v2

# Data directory structure on SOL
# /scratch/rkarumur/
# ├── data/reddit/comments/    ← .zst source files (RC_*)
# ├── data/reddit/submissions/ ← .zst source files (RS_*)
# ├── parquet_v2/              ← output .parquet files + .done.txt
# └── convert_all.py           ← the conversion script
```

### G. Key Torrent Details

```
Torrent hash: 3d426c47c767d40f82c7ef0f47c3acacedd2bf44
Location on SOL: /scratch/rkarumur/3d426c47c767d40f82c7ef0f47c3acacedd2bf44.torrent
Data directory: /scratch/rkarumur/data/reddit/
  comments/    → RC_YYYY-MM.zst files (488 files)
  submissions/ → RS_YYYY-MM.zst files
```

---

## Quick-start Checklist

1. [ ] Download Reddit comment `.zst` files via torrent to `/scratch/rkarumur/data/`
2. [ ] Wait for all downloads to complete (verify with integrity check)
3. [ ] Create conda environment: `mamba create -n reddit python=3.10 && pip install pyarrow zstandard`
4. [ ] Copy `convert_all.py` to your working directory
5. [ ] Edit `DATA_DIR` and `OUTPUT_DIR` paths in the script
6. [ ] Create and submit SLURM batch script (or run interactively for small tests)
7. [ ] Monitor progress via `tail -f` on the output log
8. [ ] If job times out or fails, just resubmit — pipeline is restart-safe
9. [ ] Verify output: count `.parquet` files, spot-check row counts and schema
10. [ ] Query with DuckDB for subreddit-specific analysis
