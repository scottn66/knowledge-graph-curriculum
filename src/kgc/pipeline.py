"""Seed → outlink graph → curriculum document."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from kgc.embed import children_from_parents, sarkar_embed
from kgc.graph import (
    articulation_points,
    condense,
    curriculum_score,
    iterative_tarjan,
    longest_path_levels,
    order_sccs,
    prepared_adjacency,
    random_walk_ppr,
)
from kgc.vault import curriculum_text, write_vault
from kgc.wiki import CrawlResult, MediaWiki, crawl, wiki_url


def sparsify_edges(
    edges: list[tuple[str, str]], ppr: dict[str, float], k: int
) -> list[tuple[str, str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for src, dst in edges:
        if dst not in grouped[src]:
            grouped[src].append(dst)
    kept: list[tuple[str, str]] = []
    for src, targets in grouped.items():
        ranked = sorted(targets, key=lambda title: (-ppr.get(title, 0.0), title))
        kept.extend((src, title) for title in ranked[:k])
    return kept


def assemble_curriculum(
    crawled: CrawlResult,
    *,
    n_walks: int = 9000,
    walk_len: int = 20,
    alpha: float = 0.15,
    sparsify_k: int = 0,
    tau: float = 1.0,
    rng_seed: int = 0,
) -> dict:
    """Run SCC, levels, PPR, and the disk embedding on a finished crawl."""
    import random

    titles = sorted(crawled.depth_of)
    edges = list(crawled.edges)
    if sparsify_k:
        adj = prepared_adjacency(titles, edges)
        preliminary = random_walk_ppr(
            titles,
            adj,
            crawled.seed,
            n_walks=n_walks,
            walk_len=walk_len,
            alpha=alpha,
            rng=random.Random(rng_seed),
        )
        edges = sparsify_edges(edges, preliminary, sparsify_k)

    adj = prepared_adjacency(titles, edges)
    raw_sccs = iterative_tarjan(titles, adj)
    sccs = order_sccs(raw_sccs)
    membership, dag_edges = condense(titles, edges, sccs)
    scc_levels = longest_path_levels(len(sccs), dag_edges)
    ppr = random_walk_ppr(
        titles,
        adj,
        crawled.seed,
        n_walks=n_walks,
        walk_len=walk_len,
        alpha=alpha,
        rng=random.Random(rng_seed),
    )
    gateways = set(articulation_points(titles, edges))
    children = children_from_parents(crawled.seed, crawled.parent)
    # A sparsified graph can drop the discovery edge. Reattach orphans to the seed
    # so the embedding still places every article.
    placed = set(children)
    for title in titles:
        if title not in placed:
            children[title] = []
    reachable = set()

    def mark(node: str) -> None:
        if node in reachable:
            return
        reachable.add(node)
        for child in children.get(node, ()):
            mark(child)

    mark(crawled.seed)
    for title in titles:
        if title not in reachable:
            children.setdefault(crawled.seed, []).append(title)
            children[title] = children.get(title, [])
    for node in list(children):
        children[node] = sorted(set(children[node]))
    positions = sarkar_embed(crawled.seed, children, tau=tau)

    out_degree = {title: 0 for title in titles}
    in_degree = {title: 0 for title in titles}
    tree_edges = set()
    for title, par in crawled.parent.items():
        if par:
            tree_edges.add((par, title))
    clean_edges = []
    seen = set()
    for src, dst in edges:
        if src not in crawled.depth_of or dst not in crawled.depth_of or src == dst:
            continue
        if (src, dst) in seen:
            continue
        seen.add((src, dst))
        out_degree[src] += 1
        in_degree[dst] += 1
        clean_edges.append((src, dst))

    scc_members: list[list[str]] = []
    for component in sccs:
        ranked = sorted(component, key=lambda title: (-ppr[title], title))
        scc_members.append(ranked)

    nodes = []
    for title in titles:
        scc_id = membership[title]
        level = scc_levels[scc_id]
        score = curriculum_score(ppr[title], level)
        point = positions.get(title, 0j)
        nodes.append({
            "id": title,
            "title": title,
            "level": level,
            "bfs_depth": crawled.depth_of.get(title, 0),
            "ppr": round(ppr[title], 8),
            "score": round(score, 8),
            "scc_id": scc_id,
            "scc_size": len(sccs[scc_id]),
            "out_degree": out_degree[title],
            "in_degree": in_degree[title],
            "gateway": title in gateways,
            "x": round(point.real, 6),
            "y": round(point.imag, 6),
            "wikipedia": wiki_url(title),
        })
    nodes.sort(key=lambda item: (item["level"], -item["score"], item["title"]))

    scc_docs = []
    for scc_id, members in enumerate(scc_members):
        scc_docs.append({
            "id": scc_id,
            "size": len(members),
            "level": scc_levels[scc_id],
            "members": members,
            "ppr": round(sum(ppr[member] for member in members), 8),
        })

    level_counts: dict[str, int] = {}
    for node in nodes:
        key = str(node["level"])
        level_counts[key] = level_counts.get(key, 0) + 1
    multi = [scc for scc in scc_docs if scc["size"] > 1]
    max_level = max(scc_levels) if scc_levels else 0
    mass_to_3 = sum(
        ppr[title] for title in titles if scc_levels[membership[title]] <= 3
    )
    if abs(mass_to_3 - 1.0) < 1e-9:
        mass_to_3 = 1.0
    node_count = len(nodes)
    edge_count = len(clean_edges)
    stats = {
        "nodes": node_count,
        "edges": edge_count,
        "scc_count": len(scc_docs),
        "singleton_sccs": len(scc_docs) - len(multi),
        "multi_sccs": len(multi),
        "largest_scc": max((scc["size"] for scc in scc_docs), default=0),
        "max_level": max_level,
        "level_counts": level_counts,
        "dag_edges": len(dag_edges),
        "compression": round(1 - (len(scc_docs) / node_count), 6) if node_count else 0,
        "gateways": len(gateways),
        "ppr_mass_to_level_3": round(mass_to_3, 8),
        "avg_out_degree": round(edge_count / node_count, 6) if node_count else 0,
    }
    params = {
        "depth": crawled.params.get("depth"),
        "fanout": crawled.params.get("fanout"),
        "max_nodes": crawled.params.get("max_nodes"),
        "walks": n_walks,
        "walk_len": walk_len,
        "alpha": alpha,
        "sparsify_k": sparsify_k,
        "tau": tau,
        "rng_seed": rng_seed,
        "source": "en.wikipedia.org mainspace outlinks in prose order",
        "edge_meaning": "u → v means Wikipedia page u links to v",
        "score": "PPR / (1 + topological level)",
    }
    return {
        "seed": crawled.seed,
        "requested_seed": crawled.requested_seed,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "params": params,
        "stats": stats,
        "nodes": nodes,
        "edges": [
            {"source": src, "target": dst, "tree": (src, dst) in tree_edges}
            for src, dst in clean_edges
        ],
        "sccs": scc_docs,
        "dag_edges": [{"source": src, "target": dst} for src, dst in dag_edges],
    }


def run_pipeline(
    seed: str,
    *,
    depth: int = 2,
    fanout: int = 15,
    max_nodes: int = 150,
    n_walks: int = 9000,
    walk_len: int = 20,
    alpha: float = 0.15,
    sparsify_k: int = 0,
    tau: float = 1.0,
    rng_seed: int = 0,
    vault: str | Path | None = None,
    web_json: str | Path | None = None,
    curriculum_txt: str | Path | None = None,
    wiki: MediaWiki | None = None,
    on_progress=None,
) -> dict:
    crawled = crawl(
        seed,
        depth=depth,
        fanout=fanout,
        max_nodes=max_nodes,
        wiki=wiki,
        on_progress=on_progress,
    )
    document = assemble_curriculum(
        crawled,
        n_walks=n_walks,
        walk_len=walk_len,
        alpha=alpha,
        sparsify_k=sparsify_k,
        tau=tau,
        rng_seed=rng_seed,
    )
    if vault:
        write_vault(document, vault)
    payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    if web_json:
        path = Path(web_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    if curriculum_txt:
        path = Path(curriculum_txt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(curriculum_text(document), encoding="utf-8")
    return document
