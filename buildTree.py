import pandas as pd
import json

INPUT_FILE = "RC_2025-12_politics.parquet"
OUTPUT_FILE = "tree_data/RC_2025-12_politics_trees.json"

def build_comment_forest_dict(df):
    nodes = {}
    for row in df.itertuples(index=False):
        nodes[row.id] = {
            "id": row.id,
            "parent_id": row.parent_id,
            "link_id": row.link_id,
            "subreddit": row.subreddit,
            "author": row.author,
            "body": row.body,
            "score": row.score,
            "controversiality": row.controversiality,
            "created_utc": row.created_utc,
            "children": []
        }

    posts = {}

    for row in df.itertuples(index=False):
        node = nodes[row.id]
        parent_key = row.parent_id.split("_", 1)[-1]
        if parent_key in nodes:
            nodes[parent_key]["children"].append(node)
        else:
            link_key = row.link_id.split("_", 1)[-1]
            if link_key not in posts:
                posts[link_key] = []
            posts[link_key].append(node)

    return posts


df = pd.read_parquet(INPUT_FILE)
posts = build_comment_forest_dict(df)

with open(OUTPUT_FILE, "w") as f:
    json.dump(posts, f)
