"""Obsidian vault: one note per concept, one note per multi-node SCC."""

from __future__ import annotations

import json
import re
from pathlib import Path


def safe_filename(title: str) -> str:
    cleaned = title
    for char in '\\/:*?"<>|':
        cleaned = cleaned.replace(char, " - ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or "untitled"


def _yaml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _link_name(title: str, names: dict[str, str]) -> str:
    return names[title]


def write_vault(doc: dict, directory: str | Path) -> Path:
    """Replace generated notes in `directory`. Returns the directory path."""
    root = Path(directory)
    concepts = root / "concepts"
    groups = root / "SCC Groups"
    if concepts.exists():
        for path in concepts.glob("*.md"):
            path.unlink()
    if groups.exists():
        for path in groups.glob("*.md"):
            path.unlink()
    concepts.mkdir(parents=True, exist_ok=True)
    groups.mkdir(parents=True, exist_ok=True)

    names: dict[str, str] = {}
    used: set[str] = set()
    for node in sorted(doc["nodes"], key=lambda item: item["title"]):
        base = safe_filename(node["title"])
        candidate = base
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{base} {suffix}"
            suffix += 1
        used.add(candidate.casefold())
        names[node["title"]] = candidate

    by_title = {node["title"]: node for node in doc["nodes"]}
    outgoing: dict[str, list[str]] = {node["title"]: [] for node in doc["nodes"]}
    incoming: dict[str, list[str]] = {node["title"]: [] for node in doc["nodes"]}
    for edge in doc["edges"]:
        outgoing[edge["source"]].append(edge["target"])
        incoming[edge["target"]].append(edge["source"])

    scc_by_id = {scc["id"]: scc for scc in doc["sccs"]}
    members_of = {}
    for scc in doc["sccs"]:
        for member in scc["members"]:
            members_of[member] = scc

    group_names: dict[int, str] = {}
    for scc in doc["sccs"]:
        if scc["size"] < 2:
            continue
        label = scc["members"][0]
        base = safe_filename(f"SCC {scc['id']} — {label}")
        candidate = base
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{base} {suffix}"
            suffix += 1
        used.add(candidate.casefold())
        group_names[scc["id"]] = candidate

    def concept_link(title: str) -> str:
        return f"[[{_link_name(title, names)}|{title}]]" if names[title] != title else f"[[{title}]]"

    def scc_link(scc_id: int) -> str:
        scc = scc_by_id[scc_id]
        if scc["size"] < 2:
            return concept_link(scc["members"][0])
        stem = group_names[scc_id]
        pretty = f"SCC {scc_id} — {scc['members'][0]}"
        return f"[[{stem}|{pretty}]]"

    for node in doc["nodes"]:
        peers = [
            member
            for member in members_of[node["title"]]["members"]
            if member != node["title"]
        ]
        lines = [
            "---",
            f"title: {_yaml_string(node['title'])}",
            f"topo_level: {node['level']}",
            f"ppr: {node['ppr']:.8f}",
            f"curriculum_score: {node['score']:.8f}",
            f"scc_id: {node['scc_id']}",
            f"scc_size: {node['scc_size']}",
            f"out_degree: {node['out_degree']}",
            f"in_degree: {node['in_degree']}",
            f"bfs_depth: {node['bfs_depth']}",
            f"gateway: {'true' if node['gateway'] else 'false'}",
            f"wikipedia: {_yaml_string(node['wikipedia'])}",
            "---",
            "",
            f"# {node['title']}",
            "",
            f"Curriculum score {node['score']:.6f} · level {node['level']} · "
            f"PPR {node['ppr']:.6f}.",
            "",
            "## Links to",
        ]
        outs = outgoing[node["title"]]
        if outs:
            lines.extend(f"- {concept_link(title)}" for title in outs)
        else:
            lines.append("- None in this crawl.")
        lines.extend(["", "## Linked from"])
        inns = incoming[node["title"]]
        if inns:
            lines.extend(f"- {concept_link(title)}" for title in inns)
        else:
            lines.append("- None in this crawl.")
        lines.extend(["", "## SCC peers"])
        if peers:
            lines.append(
                "These articles cite each other strongly enough to form one component. "
                "Neither is a prerequisite of the others inside that component."
            )
            lines.append("")
            lines.extend(f"- {concept_link(title)}" for title in peers)
        else:
            lines.append("This article is its own strongly connected component.")
        lines.append("")
        (concepts / f"{names[node['title']]}.md").write_text("\n".join(lines), encoding="utf-8")

    dag_out: dict[int, list[int]] = {scc["id"]: [] for scc in doc["sccs"]}
    dag_in: dict[int, list[int]] = {scc["id"]: [] for scc in doc["sccs"]}
    for edge in doc["dag_edges"]:
        dag_out[edge["source"]].append(edge["target"])
        dag_in[edge["target"]].append(edge["source"])

    for scc in doc["sccs"]:
        if scc["size"] < 2:
            continue
        lines = [
            "---",
            f"scc_id: {scc['id']}",
            f"scc_size: {scc['size']}",
            f"topo_level: {scc['level']}",
            f"ppr: {scc['ppr']:.8f}",
            "---",
            "",
            f"# SCC {scc['id']} — {scc['members'][0]}",
            "",
            f"{scc['size']} articles, topological level {scc['level']}, "
            f"combined PPR {scc['ppr']:.6f}.",
            "",
            "Members are mutually reachable by Wikipedia outlinks in this crawl. "
            "They are peers, not a prerequisite chain.",
            "",
            "| Rank | Concept | PPR | Curriculum score |",
            "| --- | --- | --- | --- |",
        ]
        for rank, title in enumerate(scc["members"], start=1):
            node = by_title[title]
            lines.append(
                f"| {rank} | {concept_link(title)} | {node['ppr']:.6f} | {node['score']:.6f} |"
            )
        lines.extend(["", "## Feeds into"])
        if dag_out[scc["id"]]:
            lines.extend(f"- {scc_link(target)}" for target in dag_out[scc["id"]])
        else:
            lines.append("- Nothing further in this crawl.")
        lines.extend(["", "## Fed by"])
        if dag_in[scc["id"]]:
            lines.extend(f"- {scc_link(source)}" for source in dag_in[scc["id"]])
        else:
            lines.append("- This component is a source of the condensation.")
        lines.append("")
        (groups / f"{group_names[scc['id']]}.md").write_text("\n".join(lines), encoding="utf-8")

    seed = doc["seed"]
    params = doc["params"]
    stats = doc["stats"]
    moc = [
        "---",
        f"seed: {_yaml_string(seed)}",
        f"nodes: {stats['nodes']}",
        f"edges: {stats['edges']}",
        f"scc_count: {stats['scc_count']}",
        "---",
        "",
        f"# MOC — {seed}",
        "",
        "Reading order for a newcomer, built from the English Wikipedia outlink graph.",
        "",
        f"- Seed: {concept_link(seed)}",
        f"- Crawl: depth {params['depth']}, fanout {params['fanout']}, "
        f"max {params['max_nodes']} articles",
        f"- Graph: {stats['nodes']} articles, {stats['edges']} directed links, "
        f"average out-degree {stats['avg_out_degree']:.2f}",
        f"- SCCs: {stats['scc_count']} ({stats['singleton_sccs']} singleton, "
        f"{stats['multi_sccs']} multi-article), largest {stats['largest_scc']}",
        f"- Condensation: {stats['scc_count']} super-nodes, {stats['dag_edges']} DAG edges "
        f"({stats['compression'] * 100:.1f}% fewer nodes)",
        f"- Deepest level: {stats['max_level']}",
        f"- Gateways (undirected cut vertices): {stats['gateways']}",
        "",
        "An edge A → B means A links to B. Level 0 components are not linked to "
        "from any other component. Read earlier levels first. Inside a level, "
        "higher CurriculumScore = PPR / (1 + level) comes first. "
        "A ⟲SCC(n) tag means the article sits in a mutual-reference cluster of size n.",
        "",
    ]
    levels: dict[int, list[dict]] = {}
    for node in doc["nodes"]:
        levels.setdefault(node["level"], []).append(node)
    for level in sorted(levels):
        moc.append(f"## Level {level}")
        moc.append("")
        group = sorted(levels[level], key=lambda item: (-item["score"], item["title"]))
        for node in group:
            tag = f" ⟲SCC({node['scc_size']})" if node["scc_size"] > 1 else ""
            gateway = " · gateway" if node["gateway"] else ""
            moc.append(
                f"- {concept_link(node['title'])}{tag} · score {node['score']:.6f} · "
                f"PPR {node['ppr']:.6f}{gateway}"
            )
        moc.append("")
    (root / f"MOC — {safe_filename(seed)}.md").write_text("\n".join(moc), encoding="utf-8")

    dag_lines = [
        f"# Condensation DAG — {seed}",
        "",
        "Each super-node is one strongly connected component. "
        "An arrow is a Wikipedia link from one component to another, with the cycle collapsed.",
        "",
        "| SCC | Size | Level | Representative | Combined PPR |",
        "| --- | --- | --- | --- | --- |",
    ]
    for scc in sorted(doc["sccs"], key=lambda item: (item["level"], -item["size"], item["id"])):
        dag_lines.append(
            f"| {scc['id']} | {scc['size']} | {scc['level']} | "
            f"{concept_link(scc['members'][0])} | {scc['ppr']:.6f} |"
        )
    dag_lines.extend(["", "## Edges", ""])
    if doc["dag_edges"]:
        for edge in doc["dag_edges"]:
            dag_lines.append(f"- {scc_link(edge['source'])} → {scc_link(edge['target'])}")
    else:
        dag_lines.append("- The condensation has no edges.")
    dag_lines.append("")
    (root / "Condensation DAG.md").write_text("\n".join(dag_lines), encoding="utf-8")

    (root / "graph_data.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return root


def curriculum_text(doc: dict) -> str:
    stats = doc["stats"]
    params = doc["params"]
    lines = [
        f"Curriculum for {doc['seed']}",
        f"depth {params['depth']} · fanout {params['fanout']} · max {params['max_nodes']}",
        f"nodes {stats['nodes']} · edges {stats['edges']} · SCCs {stats['scc_count']} "
        f"({stats['multi_sccs']} multi, largest {stats['largest_scc']}) · "
        f"max level {stats['max_level']}",
        "CurriculumScore(v) = PPR(v) / (1 + level(v))",
        "",
    ]
    levels: dict[int, list[dict]] = {}
    for node in doc["nodes"]:
        levels.setdefault(node["level"], []).append(node)
    for level in sorted(levels):
        lines.append(f"LEVEL {level} ({len(levels[level])} articles)")
        group = sorted(levels[level], key=lambda item: (-item["score"], item["title"]))
        for rank, node in enumerate(group, start=1):
            cluster = f" SCC({node['scc_size']})" if node["scc_size"] > 1 else ""
            lines.append(
                f"{rank:4d}  {node['score']:.6f}  {node['ppr']:.6f}  "
                f"out {node['out_degree']:<3d}{cluster}  {node['title']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
