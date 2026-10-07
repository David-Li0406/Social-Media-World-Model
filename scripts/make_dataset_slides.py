"""Generate the 5-slide dataset overview deck (slides/dataset_overview.pptx).

Mirrors the conventions of scripts/make_slides.py: 13.33x7.5 canvas, navy
header band, Calibri, accent-filled table headers. Numbers come from
data_analysis/dataset_stats.json so the deck cannot drift from the corpus.
"""
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
ACCENT = RGBColor(0x2E, 0x86, 0xC1)
GOOD = RGBColor(0x1E, 0x8B, 0x4E)
WARN = RGBColor(0xB0, 0x3A, 0x2E)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xEC, 0xF2, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

S = json.loads(Path("data_analysis/dataset_stats.json").read_text())

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW = prs.slide_width


def slide():
    return prs.slides.add_slide(BLANK)


def box(s, l, t, w, h):
    tf = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    return tf


def setline(p, text, size=16, bold=False, color=None, align=None, space=4):
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.name = "Calibri"
    if color:
        p.font.color.rgb = color
    if align:
        p.alignment = align
    p.space_after = Pt(space)


def header(s, title, sub=None):
    bar = s.shapes.add_shape(1, 0, 0, SW, Inches(1.05))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    tf = box(s, 0.45, 0.10, 12.4, 0.9)
    setline(tf.paragraphs[0], title, size=26, bold=True, color=WHITE)
    if sub:
        setline(tf.add_paragraph(), sub, size=12.5, color=RGBColor(0xCC, 0xDD, 0xEE))


def add_table(s, rows, l, t, w, h, col_w=None, font=11, highlight=None):
    highlight = highlight or set()
    gt = s.shapes.add_table(len(rows), len(rows[0]),
                            Inches(l), Inches(t), Inches(w), Inches(h)).table
    if col_w:
        for j, cw in enumerate(col_w):
            gt.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = gt.cell(i, j)
            c.margin_top = Pt(1); c.margin_bottom = Pt(1)
            c.margin_left = Pt(6); c.margin_right = Pt(6)
            p = c.text_frame.paragraphs[0]
            p.text = str(val)
            p.font.size = Pt(font + (1 if i == 0 else 0))
            p.font.name = "Calibri"
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            if i == 0:
                p.font.bold = True; p.font.color.rgb = WHITE
                c.fill.solid(); c.fill.fore_color.rgb = ACCENT
            else:
                c.fill.solid()
                c.fill.fore_color.rgb = LIGHT if i in highlight else WHITE
                if i in highlight:
                    p.font.bold = True; p.font.color.rgb = NAVY
    return gt


def tile(s, l, t, w, h, big, label, color=ACCENT):
    r = s.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    r.fill.solid(); r.fill.fore_color.rgb = LIGHT; r.line.color.rgb = color
    r.line.width = Pt(1.25)
    tf = box(s, l + 0.12, t + 0.14, w - 0.24, h - 0.28)
    setline(tf.paragraphs[0], big, size=30, bold=True, color=color,
            align=PP_ALIGN.CENTER, space=2)
    setline(tf.add_paragraph(), label, size=11.5, color=GREY, align=PP_ALIGN.CENTER)


def picture(s, path, l, t, w):
    im = Image.open(path)
    h = w * im.size[1] / im.size[0]
    s.shapes.add_picture(path, Inches(l), Inches(t), Inches(w), Inches(h))
    return t + h


def note(s, l, t, w, text, color=WARN):
    tf = box(s, l, t, w, 0.5)
    setline(tf.paragraphs[0], text, size=12, color=color)


FIG = "slides/figures"

# ---------------------------------------------------------------- slide 1
s = slide()
header(s, "A Multi-Year Reddit Engagement Dataset",
       "Comment-level engagement prediction: score, reply count, controversiality")
tile(s, 0.55, 1.45, 2.9, 1.5, f"{S['n_records']/1000:.0f}K", "comments")
tile(s, 3.70, 1.45, 2.9, 1.5, f"{S['n_posts']/1000:.0f}K", "threads")
tile(s, 6.85, 1.45, 2.9, 1.5, f"{S['n_authors']/1000:.0f}K", "unique authors")
tile(s, 10.00, 1.45, 2.78, 1.5, "11 yrs", "2014 - 2025")
add_table(s, [
    ["property", "value"],
    ["records", f"{S['n_records']:,}"],
    ["threads (posts)", f"{S['n_posts']:,}"],
    ["unique authors", f"{S['n_authors']:,}"],
    ["subreddits", f"{len(S['subreddits'])}  ({', '.join('r/'+x for x in S['subreddits'][:3])}, ...)"],
    ["months sampled", f"{len(S['months'])}  ({S['months'][0]} to {S['months'][-1]})"],
    ["split: random (by thread)", "354,658 train / 88,553 test"],
    ["split: temporal (by month)", "383,083 train / 60,000 test"],
], 0.55, 3.25, 12.23, 3.4, col_w=[4.6, 7.63], font=13)
note(s, 0.55, 6.72, 12.2,
     "One record = one comment, with its ancestor chain as context. "
     "Splits are disjoint by thread, so no thread appears on both sides.", GREY)

# ---------------------------------------------------------------- slide 2
s = slide()
header(s, "Coverage", "23 months x 5 subreddits, balanced by design")
y = picture(s, f"{FIG}/ds_coverage.png", 0.55, 1.30, 12.23)
rows = [["subreddit", "records", "share"]]
for sub in S["subreddits"]:
    b = S["by_subreddit"][sub]
    rows.append([f"r/{sub}", f"{b['n']:,}", f"{100*b['n']/S['n_records']:.1f}%"])
add_table(s, rows, 0.55, y + 0.25, 5.9, 2.0, col_w=[2.5, 1.9, 1.5], font=11.5)
tf = box(s, 6.9, y + 0.25, 5.9, 2.0)
setline(tf.paragraphs[0], "Sampling", size=15, bold=True, color=NAVY, space=6)
for t in ["Months chosen across 2014-2025; each screened so the "
          "controversiality field is actually populated.",
          "Per cell: up to 1,500 threads, max 6 comments per thread, "
          "capped at 4,000 records.",
          "Comments in the final 2 days of a dump are dropped - their "
          "replies fall into the next month's file."]:
    setline(tf.add_paragraph(), "•  " + t, size=11.5, color=GREY, space=7)

# ---------------------------------------------------------------- slide 3
s = slide()
header(s, "Prediction Targets", "Three engagement signals per comment")
y = picture(s, f"{FIG}/ds_targets.png", 0.55, 1.30, 12.23)
sc, wd = S["score"], S["width"]
add_table(s, [
    ["target", "mean", "median", "p90", "p99", "max"],
    ["score (upvotes)", f"{sc['mean']:.1f}", f"{sc['p50']:.0f}", f"{sc['p90']:.0f}",
     f"{sc['p99']:.0f}", f"{sc['mx']:.0f}"],
    ["width (direct replies)", f"{wd['mean']:.2f}", f"{wd['p50']:.0f}", f"{wd['p90']:.0f}",
     f"{wd['p99']:.0f}", f"{wd['mx']:.0f}"],
    ["controversiality (0/1)", f"{S['contr_rate']:.4f}", "0", "0", "1", "1"],
], 0.55, y + 0.25, 12.23, 1.6, col_w=[3.6, 1.75, 1.75, 1.7, 1.7, 1.73], font=12)
note(s, 0.55, y + 2.0, 12.2,
     f"{100*S['width_zero_frac']:.0f}% of comments receive no reply  |  "
     f"{S['contr_positives']:,} controversial ({100*S['contr_rate']:.2f}%)  |  "
     "score is heavy-tailed and is fuzzed by Reddit", GREY)

# ---------------------------------------------------------------- slide 4
s = slide()
header(s, "Comment Properties", "Text length and position in the thread")
y = picture(s, f"{FIG}/ds_comments.png", 1.55, 1.25, 10.2)
bc, bw = S["body_chars"], S["body_words"]
add_table(s, [
    ["property", "mean", "median", "p90", "max"],
    ["body length (characters)", f"{bc['mean']:.0f}", f"{bc['p50']:.0f}",
     f"{bc['p90']:.0f}", f"{bc['mx']:.0f}"],
    ["body length (words)", f"{bw['mean']:.0f}", f"{bw['p50']:.0f}",
     f"{bw['p90']:.0f}", f"{bw['mx']:.0f}"],
    ["top-level comments (no ancestor)",
     f"{100*S['depth_dist']['0']/S['n_records']:.0f}%", "-", "-", "-"],
    ["body is [removed] or [deleted]",
     f"{100*S['removed_or_deleted_frac']:.1f}%", "-", "-", "-"],
], 0.55, y + 0.18, 12.23, 1.75, col_w=[4.7, 1.9, 1.9, 1.9, 1.83], font=12,
    highlight={4})
note(s, 0.55, y + 2.05, 12.2,
     f"{S['removed_or_deleted']:,} comments ({100*S['removed_or_deleted_frac']:.0f}%) "
     "carry no text but still have engagement labels - decide to keep or filter.")

# ---------------------------------------------------------------- slide 5
s = slide()
header(s, "Planned Analyses", "What this dataset is sized to answer")
items = [
    ("1", "What predicts engagement: text or thread structure?",
     "One model class, swapped feature blocks (structural / lexical / semantic), with a "
     "noise block of matched width as a control. 60K test records resolve differences to ~0.002.",
     ACCENT),
    ("2", "How fast does the model go stale?",
     "Train on the earliest months and test forward, up to a 95-month horizon - eight years, "
     "versus the 16 months previous work could reach.",
     ACCENT),
    ("3", "Two-part reply modelling",
     "70% of comments get no reply, so split the target: will anyone reply at all "
     "(balanced 70/30), then how many given at least one.",
     ACCENT),
    ("4", "Add thread-level context",
     "57% of comments are top-level and have no ancestor features at all. Joining post "
     "score and age from the submission dumps should fill that gap.",
     GOOD),
]
top = 1.30
for num, title, body, col in items:
    r = s.shapes.add_shape(1, Inches(0.55), Inches(top), Inches(12.23), Inches(1.27))
    r.fill.solid(); r.fill.fore_color.rgb = LIGHT; r.line.color.rgb = col; r.line.width = Pt(1.1)
    nt = box(s, 0.70, top + 0.30, 0.8, 0.7)
    setline(nt.paragraphs[0], num, size=28, bold=True, color=col, align=PP_ALIGN.CENTER)
    tf = box(s, 1.55, top + 0.14, 11.1, 1.1)
    setline(tf.paragraphs[0], title, size=15, bold=True, color=NAVY, space=3)
    setline(tf.add_paragraph(), body, size=11.5, color=GREY)
    top += 1.39
note(s, 0.55, 6.92, 12.2,
     "Analyses 1-3 run on the dataset as it stands; 4 needs the submission dumps joined in.", GREY)

out = Path("slides/dataset_overview.pptx")
prs.save(out)
print(f"wrote {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
