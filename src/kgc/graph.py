"""Iterative Tarjan SCC, condensation, longest-path levels, PPR, gateways.

Edge direction is the Wikipedia outlink: u → v means page u links to v.
Level 0 components are condensation sources (in-degree 0). CurriculumScore
is PPR(v) / (1 + level(v)).
"""

from __future__ import annotations

import random
from collections import defaultdict


def iterative_tarjan(nodes: list[str], adj: dict[str, list[str]]) -> list[list[str]]:
    """Tarjan SCCs. `nodes` is the visit order; adjacency entries must be in `nodes`."""
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    sccs: list[list[str]] = []
    counter = 0

    def strongconnect(start: str) -> None:
        nonlocal counter
        index[start] = lowlink[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        work: list[tuple[str, int]] = [(start, 0)]
        while work:
            node, ni = work[-1]
            neighbors = adj.get(node, ())
            if ni < len(neighbors):
                work[-1] = (node, ni + 1)
                nxt = neighbors[ni]
                if nxt not in index:
                    index[nxt] = lowlink[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, 0))
                elif nxt in on_stack:
                    lowlink[node] = min(lowlink[node], index[nxt])
            else:
                if lowlink[node] == index[node]:
                    component: list[str] = []
                    while True:
                        popped = stack.pop()
                        on_stack.remove(popped)
                        component.append(popped)
                        if popped == node:
                            break
                    sccs.append(component)
                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])

    for node in nodes:
        if node not in index:
            strongconnect(node)
    return sccs


def prepared_adjacency(
    nodes: list[str], edges: list[tuple[str, str]]
) -> dict[str, list[str]]:
    present = set(nodes)
    raw: dict[str, set[str]] = {node: set() for node in nodes}
    for src, dst in edges:
        if src in present and dst in present and src != dst:
            raw[src].add(dst)
    return {node: sorted(raw[node]) for node in nodes}


def condense(
    nodes: list[str], edges: list[tuple[str, str]], sccs: list[list[str]]
) -> tuple[dict[str, int], list[tuple[int, int]]]:
    membership = {}
    for scc_id, component in enumerate(sccs):
        for node in component:
            membership[node] = scc_id
    dag: set[tuple[int, int]] = set()
    for src, dst in edges:
        if src not in membership or dst not in membership:
            continue
        a = membership[src]
        b = membership[dst]
        if a != b:
            dag.add((a, b))
    return membership, sorted(dag)


def longest_path_levels(scc_count: int, dag_edges: list[tuple[int, int]]) -> list[int]:
    """Level of each SCC: longest path from any in-degree-0 source."""
    outgoing: dict[int, list[int]] = defaultdict(list)
    indeg = [0] * scc_count
    for src, dst in dag_edges:
        outgoing[src].append(dst)
        indeg[dst] += 1
    queue = [i for i in range(scc_count) if indeg[i] == 0]
    order: list[int] = []
    head = 0
    work = indeg[:]
    while head < len(queue):
        node = queue[head]
        head += 1
        order.append(node)
        for nxt in outgoing.get(node, ()):
            work[nxt] -= 1
            if work[nxt] == 0:
                queue.append(nxt)
    if len(order) != scc_count:
        raise RuntimeError("condensation has a cycle; Tarjan result is inconsistent")
    level = [0] * scc_count
    for node in order:
        for nxt in outgoing.get(node, ()):
            level[nxt] = max(level[nxt], level[node] + 1)
    return level


def random_walk_ppr(
    nodes: list[str],
    adj: dict[str, list[str]],
    seed: str,
    *,
    n_walks: int = 9000,
    walk_len: int = 20,
    alpha: float = 0.15,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Personalized PageRank by random walks with restart probability `alpha`."""
    if seed not in set(nodes):
        raise KeyError(seed)
    if n_walks < 1 or walk_len < 1:
        raise ValueError("walks and walk length must be positive")
    generator = rng or random.Random(0)
    counts = {node: 0 for node in nodes}
    for _ in range(n_walks):
        current = seed
        counts[current] += 1
        for _step in range(walk_len):
            neighbors = adj.get(current) or ()
            if not neighbors or generator.random() < alpha:
                current = seed
            else:
                current = generator.choice(neighbors)
            counts[current] += 1
    total = float(sum(counts.values()))
    return {node: counts[node] / total for node in nodes}


def curriculum_score(ppr: float, level: int) -> float:
    return ppr / (1 + level)


def articulation_points(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Cut vertices of the underlying undirected graph."""
    present = set(nodes)
    undirected: dict[str, set[str]] = {node: set() for node in nodes}
    for src, dst in edges:
        if src in present and dst in present and src != dst:
            undirected[src].add(dst)
            undirected[dst].add(src)
    adj = {node: sorted(undirected[node]) for node in nodes}
    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    points: set[str] = set()
    timer = 0

    for start in nodes:
        if start in discovery:
            continue
        parent[start] = None
        discovery[start] = low[start] = timer
        timer += 1
        root_children = 0
        stack = [(start, 0)]
        while stack:
            node, ni = stack[-1]
            neighbors = adj[node]
            if ni < len(neighbors):
                stack[-1] = (node, ni + 1)
                nxt = neighbors[ni]
                if nxt not in discovery:
                    parent[nxt] = node
                    discovery[nxt] = low[nxt] = timer
                    timer += 1
                    if node == start:
                        root_children += 1
                    stack.append((nxt, 0))
                elif nxt != parent.get(node):
                    low[node] = min(low[node], discovery[nxt])
            else:
                stack.pop()
                par = parent.get(node)
                if par is not None:
                    low[par] = min(low[par], low[node])
                    if parent.get(par) is not None and low[node] >= discovery[par]:
                        points.add(par)
        if root_children > 1:
            points.add(start)
    return sorted(points)


def order_sccs(sccs: list[list[str]]) -> list[list[str]]:
    """Stable ids: larger components first, then by member title."""
    return sorted(sccs, key=lambda comp: (-len(comp), min(comp)))
