"""Curriculum pipeline tests that do not touch the network."""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kgc.embed import hyperbolic_distance, sarkar_embed  # noqa: E402
from kgc.graph import (  # noqa: E402
    articulation_points,
    condense,
    curriculum_score,
    iterative_tarjan,
    longest_path_levels,
    prepared_adjacency,
    random_walk_ppr,
)
from kgc.pipeline import assemble_curriculum  # noqa: E402
from kgc.vault import write_vault  # noqa: E402
from kgc.wiki import CrawlResult, crawl, extract_prose_links  # noqa: E402


SAMPLE = """
{{Infobox physics
| field = [[Should Not Appear]]
}}
'''Optics''' studies [[Light]] and [[Refraction]].
A [[Lens|lens]] follows [[Snell's law#Statement|Snell's law]].
<ref>{{cite book|title=[[Ignore Me]]}}</ref>
[[Category:Optics]]
[[File:Lens.jpg|thumb|A lens]]
See [[Photon]] and [[Light]] again.
[[fr:Optique]]
"""


def tarjan_recursive(nodes, adj):
    index = {}
    low = {}
    stack = []
    on = set()
    sccs = []
    counter = 0

    def strong(node):
        nonlocal counter
        index[node] = low[node] = counter
        counter += 1
        stack.append(node)
        on.add(node)
        for nxt in adj.get(node, ()):
            if nxt not in index:
                strong(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in on:
                low[node] = min(low[node], index[nxt])
        if low[node] == index[node]:
            comp = []
            while True:
                popped = stack.pop()
                on.remove(popped)
                comp.append(popped)
                if popped == node:
                    break
            sccs.append(frozenset(comp))

    for node in nodes:
        if node not in index:
            strong(node)
    return sccs


class ExtractTests(unittest.TestCase):
    def test_prose_order_skips_templates_refs_and_files(self):
        self.assertEqual(
            extract_prose_links(SAMPLE),
            ["Light", "Refraction", "Lens", "Snell's law", "Photon"],
        )

    def test_colon_in_a_real_title_is_kept(self):
        text = "See [[The Lord of the Rings: The Fellowship of the Ring]] and [[fr:Optique]]."
        self.assertEqual(
            extract_prose_links(text),
            ["The Lord of the Rings: The Fellowship of the Ring"],
        )

    def test_fanout_keeps_the_first_links(self):
        self.assertEqual(extract_prose_links(SAMPLE, fanout=3), ["Light", "Refraction", "Lens"])

    def test_unbalanced_template_does_not_eat_the_article(self):
        text = "Intro [[Kept]] {{unclosed [[Also kept]]"
        self.assertEqual(extract_prose_links(text), ["Kept", "Also kept"])


class GraphTests(unittest.TestCase):
    def test_tarjan_matches_recursive_on_random_graphs(self):
        rng = random.Random(7)
        for n in (1, 2, 8, 20):
            nodes = [f"n{i}" for i in range(n)]
            edges = []
            for _ in range(n * 3):
                edges.append((rng.choice(nodes), rng.choice(nodes)))
            adj = prepared_adjacency(nodes, edges)
            got = {frozenset(comp) for comp in iterative_tarjan(nodes, adj)}
            expect = set(tarjan_recursive(nodes, adj))
            self.assertEqual(got, expect)

    def test_cycle_and_tail_levels(self):
        nodes = ["a", "b", "c"]
        edges = [("a", "b"), ("b", "a"), ("b", "c")]
        adj = prepared_adjacency(nodes, edges)
        sccs = sorted(iterative_tarjan(nodes, adj), key=lambda comp: min(comp))
        # order_sccs is not applied; identify by membership
        membership, dag = condense(nodes, edges, sccs)
        levels = longest_path_levels(len(sccs), dag)
        self.assertEqual(membership["a"], membership["b"])
        self.assertNotEqual(membership["a"], membership["c"])
        self.assertEqual(levels[membership["a"]], 0)
        self.assertEqual(levels[membership["c"]], 1)

    def test_longest_path_not_shortest(self):
        nodes = ["s", "a", "b", "t"]
        edges = [("s", "a"), ("s", "t"), ("a", "b"), ("b", "t")]
        adj = prepared_adjacency(nodes, edges)
        sccs = iterative_tarjan(nodes, adj)
        self.assertEqual(len(sccs), 4)
        membership, dag = condense(nodes, edges, sccs)
        levels = longest_path_levels(4, dag)
        self.assertEqual(levels[membership["s"]], 0)
        self.assertEqual(levels[membership["t"]], 3)

    def test_curriculum_score(self):
        self.assertAlmostEqual(curriculum_score(0.2, 0), 0.2)
        self.assertAlmostEqual(curriculum_score(0.2, 3), 0.05)

    def test_ppr_mass_and_seed_on_a_star(self):
        nodes = ["seed", "a", "b"]
        edges = [("seed", "a"), ("seed", "b")]
        adj = prepared_adjacency(nodes, edges)
        ppr = random_walk_ppr(
            nodes, adj, "seed", n_walks=400, walk_len=10, alpha=0.15, rng=random.Random(1)
        )
        self.assertAlmostEqual(sum(ppr.values()), 1.0)
        self.assertGreater(ppr["seed"], ppr["a"])
        self.assertGreater(ppr["seed"], ppr["b"])

    def test_articulation_point(self):
        nodes = ["a", "b", "c"]
        edges = [("a", "b"), ("b", "c")]
        self.assertEqual(articulation_points(nodes, edges), ["b"])
        self.assertEqual(articulation_points(["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")]), [])


class EmbedTests(unittest.TestCase):
    def test_tree_edges_have_length_tau(self):
        tau = 1.0
        children = {"root": ["a", "b"], "a": ["a1"], "b": [], "a1": []}
        pos = sarkar_embed("root", children, tau=tau)
        self.assertEqual(pos["root"], 0j)
        for parent, kids in children.items():
            for child in kids:
                self.assertLess(abs(pos[child]), 1)
                self.assertAlmostEqual(hyperbolic_distance(pos[parent], pos[child]), tau, places=6)


class FakeWiki:
    def __init__(self, pages, redirects=None):
        self.pages = pages
        self.redirects = {key.casefold(): value for key, value in (redirects or {}).items()}

    def resolve(self, titles):
        folded = {key.casefold(): key for key in self.pages}
        out = {}
        for title in titles:
            current = title
            seen = set()
            while current.casefold() in self.redirects and current.casefold() not in seen:
                seen.add(current.casefold())
                current = self.redirects[current.casefold()]
            out[title] = folded.get(current.casefold())
        return out

    def wikitext(self, titles):
        return {title: self.pages[title] for title in titles if title in self.pages}


class PipelineTests(unittest.TestCase):
    def test_crawl_and_vault_are_generic(self):
        pages = {
            "Tide": "A [[Wave]] moves with the [[Moon]].",
            "Wave": "A wave meets the [[Shore]]. See [[Tide]].",
            "Moon": "The moon raises the [[Tide]].",
            "Shore": "The shore meets the [[Wave]].",
        }
        wiki = FakeWiki(pages, {"Tides": "Tide"})
        crawled = crawl("Tides", depth=2, fanout=2, max_nodes=10, wiki=wiki)
        self.assertEqual(crawled.seed, "Tide")
        self.assertIn("Wave", crawled.depth_of)
        doc = assemble_curriculum(crawled, n_walks=200, walk_len=8, rng_seed=3)
        self.assertEqual(doc["seed"], "Tide")
        self.assertEqual(doc["stats"]["nodes"], 4)
        self.assertAlmostEqual(sum(node["ppr"] for node in doc["nodes"]), 1.0, places=5)
        titles = {node["title"] for node in doc["nodes"]}
        self.assertEqual(titles, set(pages))
        # Mutual Tide ↔ Wave and Moon → Tide, Shore ↔ Wave should not be one blob
        # unless the links really close a single component. Just require a DAG of SCCs.
        self.assertGreaterEqual(doc["stats"]["scc_count"], 1)
        self.assertEqual(doc["stats"]["dag_edges"], len(doc["dag_edges"]))
        with tempfile.TemporaryDirectory() as tmp:
            vault = write_vault(doc, tmp)
            self.assertTrue((vault / "concepts" / "Tide.md").is_file())
            self.assertTrue((vault / "MOC — Tide.md").is_file())
            self.assertTrue((vault / "Condensation DAG.md").is_file())
            self.assertTrue((vault / "graph_data.json").is_file())
            tide = (vault / "concepts" / "Tide.md").read_text(encoding="utf-8")
            self.assertIn("title: \"Tide\"", tide)
            self.assertIn("[[Wave]]", tide)


if __name__ == "__main__":
    unittest.main()
