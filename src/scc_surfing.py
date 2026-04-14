#!/usr/bin/env python3
"""
SCC Condensation & Topological Sort on the Surfing Knowledge Graph
==================================================================
Implements Sections 9.1–9.5 of MATHEMATICAL_FOUNDATIONS.md:
  1. Tarjan's algorithm for SCC detection
  2. Condensation DAG construction
  3. Topological level assignment
  4. PPR-weighted CurriculumScore for reading curriculum
  5. Output up to depth 3
"""

import json, math
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
# 1. Load the surfing knowledge graph
# ═══════════════════════════════════════════════════════════════

with open('/sessions/charming-exciting-cori/poincare_art_data_surfing.json') as f:
    data = json.load(f)

nodes = data['nodes']
edges = data['edges']
node_map = {n['id']: n for n in nodes}

# Build directed adjacency (forward edges only — directed graph)
adj = defaultdict(list)       # outgoing neighbors
adj_rev = defaultdict(list)   # incoming neighbors
for e in edges:
    adj[e['source']].append(e['target'])
    adj_rev[e['target']].append(e['source'])

all_ids = set(n['id'] for n in nodes)

print(f"Graph: {len(nodes)} nodes, {len(edges)} directed edges")
print(f"Avg out-degree: {len(edges)/len(nodes):.1f}")
print()

# ═══════════════════════════════════════════════════════════════
# 2. Tarjan's SCC Algorithm — O(|V| + |E|)
# ═══════════════════════════════════════════════════════════════

class TarjanSCC:
    def __init__(self, vertices, adj):
        self.adj = adj
        self.index_counter = [0]
        self.stack = []
        self.on_stack = set()
        self.index = {}
        self.lowlink = {}
        self.sccs = []

        for v in vertices:
            if v not in self.index:
                self._strongconnect(v)

    def _strongconnect(self, v):
        # Iterative Tarjan to avoid recursion limit
        work_stack = [(v, 0)]  # (node, neighbor_index)
        self.index[v] = self.lowlink[v] = self.index_counter[0]
        self.index_counter[0] += 1
        self.stack.append(v)
        self.on_stack.add(v)

        while work_stack:
            node, ni = work_stack[-1]
            neighbors = self.adj.get(node, [])

            if ni < len(neighbors):
                work_stack[-1] = (node, ni + 1)
                w = neighbors[ni]
                if w not in self.index:
                    self.index[w] = self.lowlink[w] = self.index_counter[0]
                    self.index_counter[0] += 1
                    self.stack.append(w)
                    self.on_stack.add(w)
                    work_stack.append((w, 0))
                elif w in self.on_stack:
                    self.lowlink[node] = min(self.lowlink[node], self.index[w])
            else:
                # Done with all neighbors
                if self.lowlink[node] == self.index[node]:
                    scc = []
                    while True:
                        w = self.stack.pop()
                        self.on_stack.discard(w)
                        scc.append(w)
                        if w == node:
                            break
                    self.sccs.append(scc)

                # Update parent's lowlink
                if len(work_stack) > 1:
                    parent = work_stack[-2][0]
                    self.lowlink[parent] = min(self.lowlink[parent], self.lowlink[node])
                work_stack.pop()


tarjan = TarjanSCC(all_ids, adj)
sccs = tarjan.sccs

print(f"═══ SCC ANALYSIS ═══")
print(f"Total SCCs found: {len(sccs)}")

# Classify SCCs by size
singletons = [s for s in sccs if len(s) == 1]
multi = [s for s in sccs if len(s) > 1]
multi.sort(key=len, reverse=True)

print(f"  Singleton SCCs (size 1): {len(singletons)}")
print(f"  Multi-node SCCs (size > 1): {len(multi)}")
if multi:
    print(f"  Largest SCC: {len(multi[0])} nodes")
    for i, scc in enumerate(multi[:5]):
        names = [node_map[nid]['name'] for nid in scc[:8]]
        suffix = f" + {len(scc)-8} more" if len(scc) > 8 else ""
        print(f"    SCC-{i}: [{', '.join(names)}{suffix}]")
print()

# ═══════════════════════════════════════════════════════════════
# 3. Condensation DAG
# ═══════════════════════════════════════════════════════════════

# Map each node to its SCC index
node_to_scc = {}
for i, scc in enumerate(sccs):
    for nid in scc:
        node_to_scc[nid] = i

# Build condensation DAG edges
cdag_adj = defaultdict(set)
cdag_adj_rev = defaultdict(set)
for e in edges:
    s_scc = node_to_scc[e['source']]
    t_scc = node_to_scc[e['target']]
    if s_scc != t_scc:
        cdag_adj[s_scc].add(t_scc)
        cdag_adj_rev[t_scc].add(s_scc)

cdag_edges = sum(len(v) for v in cdag_adj.values())
print(f"═══ CONDENSATION DAG ═══")
print(f"Super-nodes: {len(sccs)}")
print(f"DAG edges: {cdag_edges}")
print(f"Compression: {len(nodes)} nodes → {len(sccs)} super-nodes ({100*(1-len(sccs)/len(nodes)):.1f}% reduction)")
print()

# ═══════════════════════════════════════════════════════════════
# 4. Topological Level Assignment
# ═══════════════════════════════════════════════════════════════

# BFS from sources (in-degree 0 in condensation DAG)
scc_in_degree = defaultdict(int)
for i in range(len(sccs)):
    for j in cdag_adj.get(i, set()):
        scc_in_degree[j] += 1

# Source SCCs (level 0): no incoming edges in condensation
sources = [i for i in range(len(sccs)) if scc_in_degree[i] == 0]

# BFS to assign levels
from collections import deque
level = {}
queue = deque()
for s in sources:
    level[s] = 0
    queue.append(s)

while queue:
    cur = queue.popleft()
    for nbr in cdag_adj.get(cur, set()):
        new_level = level[cur] + 1
        if nbr not in level or new_level > level[nbr]:
            level[nbr] = new_level
            queue.append(nbr)

# Assign level to each node
node_level = {}
for nid in all_ids:
    scc_idx = node_to_scc[nid]
    node_level[nid] = level.get(scc_idx, 0)

max_level = max(node_level.values()) if node_level else 0

print(f"═══ TOPOLOGICAL LEVELS ═══")
print(f"Maximum depth: {max_level}")
for lv in range(max_level + 1):
    nodes_at_level = [nid for nid, l in node_level.items() if l == lv]
    print(f"  Level {lv}: {len(nodes_at_level)} nodes")
print()

# ═══════════════════════════════════════════════════════════════
# 5. Reading Curriculum — Depth ≤ 3, PPR-ranked
# ═══════════════════════════════════════════════════════════════

print(f"═══ READING CURRICULUM (Depth ≤ 3) ═══")
print(f"CurriculumScore(v) = PPR(v) / (1 + level(v))")
print()

for lv in range(min(max_level + 1, 4)):  # levels 0, 1, 2, 3
    nodes_at_level = []
    for nid, l in node_level.items():
        if l == lv:
            n = node_map[nid]
            ppr = n.get('ppr', 0)
            score = ppr / (1 + lv)
            nodes_at_level.append({
                'id': nid,
                'name': n['name'],
                'domain': n['domain'],
                'ppr': ppr,
                'degree': n.get('degree', 0),
                'score': score,
                'scc_size': len(sccs[node_to_scc[nid]]),
            })

    # Sort by CurriculumScore descending
    nodes_at_level.sort(key=lambda x: x['score'], reverse=True)

    # Level header
    if lv == 0:
        label = "BASIS CONCEPTS (no prerequisites)"
    elif lv == 1:
        label = "FIRST-ORDER (requires level 0)"
    elif lv == 2:
        label = "SECOND-ORDER (requires levels 0-1)"
    else:
        label = "THIRD-ORDER (requires levels 0-2)"

    print(f"{'─'*70}")
    print(f"LEVEL {lv}: {label}")
    print(f"{'─'*70}")
    print(f"{'Rank':<5} {'CurrScore':<10} {'PPR':<10} {'Deg':<5} {'SCC':<5} {'Domain':<18} {'Concept'}")
    print(f"{'─'*5} {'─'*9} {'─'*9} {'─'*4} {'─'*4} {'─'*17} {'─'*30}")

    for rank, n in enumerate(nodes_at_level[:25], 1):
        scc_label = f"[{n['scc_size']}]" if n['scc_size'] > 1 else ""
        print(f"{rank:<5} {n['score']:<10.6f} {n['ppr']:<10.6f} {n['degree']:<5} {scc_label:<5} {n['domain']:<18} {n['name']}")

    if len(nodes_at_level) > 25:
        print(f"  ... and {len(nodes_at_level) - 25} more at this level")
    print()

# ═══════════════════════════════════════════════════════════════
# 6. SCC Semantic Analysis
# ═══════════════════════════════════════════════════════════════

print(f"═══ SCC SEMANTIC INTERPRETATION ═══")
print()
if multi:
    print("Multi-node SCCs (mutually referencing concept clusters):")
    for i, scc in enumerate(multi):
        names = [node_map[nid]['name'] for nid in scc]
        domains = set(node_map[nid]['domain'] for nid in scc)
        avg_ppr = sum(node_map[nid].get('ppr', 0) for nid in scc) / len(scc)
        lv = level.get(i, node_level.get(scc[0], '?'))

        print(f"  SCC-{i} (size {len(scc)}, level {node_level.get(scc[0], '?')}):")
        print(f"    Domains: {', '.join(sorted(domains))}")
        print(f"    Avg PPR: {avg_ppr:.6f}")
        print(f"    Members: {', '.join(names[:12])}")
        if len(names) > 12:
            print(f"             + {len(names)-12} more")
        print(f"    Interpretation: These concepts define each other mutually —")
        print(f"                    neither is prerequisite to the others.")
        print()
else:
    print("No multi-node SCCs found. The graph is already a DAG (no mutual reference cycles).")
    print("This means every concept has a clear directional dependency.")
    print()

# ═══════════════════════════════════════════════════════════════
# 7. Gateway Node Analysis (articulation points)
# ═══════════════════════════════════════════════════════════════

print(f"═══ GATEWAY NODES (Cut Vertices) ═══")
print(f"Nodes whose removal disconnects portions of the graph:")
print()

# Find articulation points in the undirected version
undirected = defaultdict(set)
for e in edges:
    undirected[e['source']].add(e['target'])
    undirected[e['target']].add(e['source'])

visited = set()
disc = {}
low = {}
parent = {}
ap = set()
timer = [0]

def dfs_ap(u):
    """Iterative DFS for articulation points."""
    stack = [(u, iter(undirected[u]), False)]
    visited.add(u)
    disc[u] = low[u] = timer[0]
    timer[0] += 1
    parent[u] = -1
    child_count = defaultdict(int)

    while stack:
        node, neighbors, returned = stack[-1]

        try:
            v = next(neighbors)
            if v not in visited:
                child_count[node] += 1
                visited.add(v)
                parent[v] = node
                disc[v] = low[v] = timer[0]
                timer[0] += 1
                stack.append((v, iter(undirected[v]), False))
            elif v != parent.get(node, -1):
                low[node] = min(low[node], disc[v])
        except StopIteration:
            stack.pop()
            if stack:
                par_node = stack[-1][0]
                low[par_node] = min(low[par_node], low[node])
                # Check AP condition
                if parent[par_node] == -1 and child_count[par_node] > 1:
                    ap.add(par_node)
                if parent[par_node] != -1 and low[node] >= disc[par_node]:
                    ap.add(par_node)

for nid in all_ids:
    if nid not in visited:
        dfs_ap(nid)

gateways = []
for nid in ap:
    n = node_map[nid]
    gateways.append({
        'name': n['name'],
        'domain': n['domain'],
        'ppr': n.get('ppr', 0),
        'degree': n.get('degree', 0),
        'level': node_level.get(nid, 0),
    })

gateways.sort(key=lambda x: x['ppr'], reverse=True)
for g in gateways[:15]:
    print(f"  Level {g['level']} | PPR {g['ppr']:.6f} | Deg {g['degree']:<3} | {g['domain']:<18} | {g['name']}")

print(f"\nTotal gateway nodes: {len(gateways)}")
print()

# ═══════════════════════════════════════════════════════════════
# 8. Summary Statistics
# ═══════════════════════════════════════════════════════════════

depth3_nodes = [nid for nid, l in node_level.items() if l <= 3]
depth3_ppr = sum(node_map[nid].get('ppr', 0) for nid in depth3_nodes)
total_ppr = sum(n.get('ppr', 0) for n in nodes)

print(f"═══ DEPTH-3 SUMMARY ═══")
print(f"Nodes at depth ≤ 3: {len(depth3_nodes)} / {len(nodes)} ({100*len(depth3_nodes)/len(nodes):.1f}%)")
print(f"PPR coverage at depth ≤ 3: {depth3_ppr:.4f} / {total_ppr:.4f} ({100*depth3_ppr/total_ppr:.1f}%)")
print(f"Domains represented: {len(set(node_map[nid]['domain'] for nid in depth3_nodes))}")
print(f"Multi-node SCCs: {len(multi)} (containing {sum(len(s) for s in multi)} nodes)")
print(f"Gateway nodes: {len(gateways)} (critical for graph connectivity)")
