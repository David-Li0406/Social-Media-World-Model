# A Multi-Year Reddit Engagement Dataset

Companion notes for `slides/dataset_overview.pptx`.
Regenerate both with:

```bash
PYTHONPATH=src python3 scripts/dataset_stats.py      # stats + figures
python3 scripts/make_dataset_slides.py               # deck
```

---

## 1 - Size

One record = one comment, with its ancestor chain as context.

| property | value |
|---|---:|
| records | 443,211 |
| threads (posts) | 151,249 |
| unique authors | 159,826 |
| subreddits | 5 |
| months sampled | 23 (2014-01 to 2025-09) |
| random split (by thread) | 354,658 train / 88,553 test |
| temporal split (by month) | 383,083 train / 60,000 test |

Both splits are disjoint by thread, so no thread appears on both sides.

## 2 - Coverage

23 months x 5 subreddits. Per cell: up to 1,500 threads, at most
6 comments per thread, capped at 4,000 records. Comments in the last 2 days of a
monthly dump are dropped, since their replies land in the next month's file.

| subreddit | records | share |
|---|---:|---:|
| r/Conservative | 92,000 | 20.8% |
| r/politics | 92,000 | 20.8% |
| r/worldnews | 90,402 | 20.4% |
| r/news | 88,632 | 20.0% |
| r/technology | 80,177 | 18.1% |

## 3 - Prediction targets

| target | mean | median | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| score (upvotes) | 5.1 | 1 | 9 | 58 | 15552 |
| width (direct replies) | 0.43 | 0 | 1 | 4 | 235 |
| controversiality (0/1) | 0.0416 | 0 | 0 | 1 | 1 |

- 70% of comments receive no reply
- 18,455 controversial comments (4.16%)
- score is heavy-tailed, and Reddit fuzzes the reported value

## 4 - Comment properties

| property | mean | median | p90 | max |
|---|---:|---:|---:|---:|
| body length (characters) | 221 | 91 | 626 | 11390 |
| body length (words) | 34 | 16 | 94 | 1644 |
| top-level comments (no ancestor) | 57% | - | - | - |
| body is `[removed]` / `[deleted]` | 16.8% | - | - | - |

74,329 comments carry no text but still have engagement labels -
an open decision whether to keep or filter them.

## 5 - Planned analyses

1. **What predicts engagement: text or thread structure?** One model class, swapped
   feature blocks (structural / lexical / semantic), with a noise block of matched
   width as a control. 60K test records resolve differences to ~0.002.
2. **How fast does the model go stale?** Train on the earliest months and test
   forward, up to a 95-month horizon.
3. **Two-part reply modelling.** 70% of comments get no reply, so split the
   target: will anyone reply at all, then how many given at least one.
4. **Add thread-level context.** 57% of comments are top-level and have no
   ancestor features; joining post score and age from the submission dumps should fill that gap.

Analyses 1-3 run on the dataset as it stands; 4 needs the submission dumps joined in.
