import json

INPUT_FILE = "tree_data/RC_2025-12_politics_trees.json"
OUTPUT_FILE = "chain_data/RC_2025-12_politics_chains.json"

def max_depth(node):
    if not node["children"]:
        return 1
    
    return 1 + max(max_depth(child) for child in node["children"])

def strip_children(node):
    return {
        k: v
        for k, v in node.items()
        if k != "children"
    }

def strip_solution(node):
    return {
        k: v
        for k, v in node.items()
        if k != "children" and k != "score" and k != "controversiality"
    }

def buildChain(node):
    stack = []
    stack.append((node, [strip_children(node)]))
    chains = []

    while len(stack) != 0:
        cur, context = stack.pop()

        if len(cur["children"]) == 0:
            chains.append(context)

        for child in cur["children"]:
            newCont = context + [strip_children(child)]
            stack.append((child, newCont))

    chains.sort(key=len, reverse=True)

    used_edges = set()
    selected_chains = []

    for path in chains:
        edges = [(path[i]["id"], path[i+1]["id"]) for i in range(len(path) - 1)]
        
        if any(e in used_edges for e in edges):
            continue

        if len(path) < 3:
            continue
        
        used_edges.update(edges)
        selected_chains.append(path)
    
    return selected_chains


with open(INPUT_FILE, "r") as f:
    posts = json.load(f)

dumpValue = []
for link_id, comments in posts.items():
    for comment in comments:
        dumpValue.extend([c for c in buildChain(comment) if c[0]["score"] >= 20])


with open(OUTPUT_FILE, "w") as f:
    json.dump(dumpValue, f, indent=None)
