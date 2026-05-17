import json
import random
from collections import deque


INPUT_FILE_CHAINS = "chain_data/RC_2025-12_politics_chains.json"
INPUT_FILE_TREES = "tree_data/RC_2025-12_politics_trees.json"
OUTPUT_FILE = "final_sets/RC_2025-12_politics_final_set.json"

def strip_solution(node):
    return {
        k: v
        for k, v in node.items()
        if k != "children" and k != "score" and k != "controversiality"
    }

def strip_children(node):
    return {
        k: v
        for k, v in node.items()
        if k != "children"
    }

def findNode(node, id):
    queue = deque()
    queue.append(node)

    while (len(queue) != 0):
        cur = queue.popleft()
        if (cur["id"] == id):
            return cur
        for child in cur["children"]:
            queue.append(child)

    return None

with open(INPUT_FILE_CHAINS, "r") as f:
    chains = json.load(f)

with open(INPUT_FILE_TREES, "r") as f:
    trees = json.load(f)

objects = []

for chain in chains:
    cut_point = random.randint(0, len(chain) - 2)
    context = chain[:cut_point]
    stimulus = strip_solution(chain[cut_point])

    node = None
    link_key = stimulus["link_id"].split("_", 1)[-1]
    for comment in trees[link_key]:
        if comment["id"] == chain[0]["id"]:
            node = findNode(comment, chain[cut_point]["id"])
            break

    if node == None:
        continue
    width = []
    for child in node["children"]:
        width.append(strip_children(child))
    
    objects.append({
        "context" : context,
        "stimulus" : stimulus,
        "ground_truth" : {
            "score" : chain[cut_point]["score"],
            "controversiality" : chain[cut_point]["controversiality"],
            "width" : width
        }
    })

with open(OUTPUT_FILE, "w") as f:
    json.dump(objects, f, indent=None)
    









