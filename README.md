# Knowledge Graph Curriculum

Seed any English Wikipedia article, crawl its outlink graph, and turn the cycles into a reading order.

The pipeline is generic. Optics is the worked example, captured from the live MediaWiki API on 2026-11-06.

**Demo source:** `docs/` is the GitHub Pages root (`main` / `/docs`). The project URL https://scottn66.github.io/knowledge-graph-curriculum/ is not serving yet: Pages is not enabled on the repository.

## What actually runs

| Piece | Where | State |
|---|---|---|
| Prose-order Wikipedia BFS | `src/kgc/wiki.py`, `docs/curriculum.js` | Live |
| Iterative Tarjan SCC, condensation, longest-path levels | `src/kgc/graph.py`, `docs/curriculum.js` | Live |
| Random-walk Personalized PageRank (α = 0.15, 9,000 walks, length 20) | same | Live |
| CurriculumScore = PPR / (1 + level) | same | Live |
| Sarkar tree embedding in the Poincaré disk, plus the half-plane map | `src/kgc/embed.py`, the demo | Live |
| Obsidian vault | `src/kgc/vault.py` | Live |
| Bandits, entropy, HMM/HHMM, SARSA, Dyna, counterfactuals | `theory/MATHEMATICAL_FOUNDATIONS.md` §10–19 | Not in this repo |

An edge u → v means the Wikipedia page u links to v. Templates, citations, file links, and category links are dropped before the fanout cap, so the crawl follows the article prose rather than the navigation chrome. Level 0 components are sources of the condensation (nothing else in the crawl links into them). Read earlier levels first. Inside a level, a higher score comes first.

## Optics, from the live API

Defaults: depth 2, fanout 15, 150 articles.

- 150 articles, 229 directed links
- 139 SCCs (137 singletons, 2 multi-article). Largest component: 9 (the electromagnetic-spectrum cluster). The Optics core is a component of 4: Optics, Physical optics, Ray (optics), Geometrical optics
- Levels: 4 / 17 / 47 / 82. Deepest level 3
- 14 gateways (undirected cut vertices)

The vault is `optics-vault/`: one note per article, one note per multi-article component, `MOC — Optics.md`, and `Condensation DAG.md`.

## Run

```bash
python src/build_curriculum.py --seed Optics --depth 2 --fanout 15 --max-nodes 150 \
  --vault optics-vault \
  --web-json docs/data/optics_graph.json \
  --curriculum-txt docs/optics_curriculum.txt
```

Any other title works the same way. Omit `--vault` and the notes go to `vaults/<seed>-vault/`.

```bash
python src/build_curriculum.py --seed "Tide"
python -m unittest discover -s tests -v
node tests/curriculum.test.js
```

No third-party Python packages. The web demo is static: open `docs/index.html` through a local server, or use the Pages site. The form crawls Wikipedia in the browser (`origin=*`). The saved Optics graph loads first, so the page does not call the API until you ask it to.

## Legacy synthetic pictures

`src/generate_surfing_data.py`, `src/scc_surfing.py`, and `src/render_surfing.py` build a hand-authored surfing graph and a polar plot (radius from PageRank, angle from a domain sector). Those SVGs are still in `assets/`. They are not a Wikipedia crawl, and the polar layout is not the Sarkar embedding used by the demo. `src/render_hyperbolic.py` draws the same kind of polar picture for the checked-in cognitive-science JSON.

## License

MIT.
