# Knowledge Graph Curriculum

Build a directed concept graph over any topic, then derive a **reading order** from it using classical graph theory: Tarjan's SCC → condensation DAG → topological levels → Personalized PageRank weighting. Visualize the result in the **Poincaré disk** and **upper half-plane** models of hyperbolic geometry.

**Live demo:** https://scottn66.github.io/knowledge-graph-curriculum/

## What's in the box

| Path | Purpose |
|---|---|
| `src/generate_surfing_data.py` | Build a 146-node concept graph over surfing (10 domains) + compute PPR via random walks |
| `src/scc_surfing.py` | Iterative Tarjan's SCC → condensation → BFS topological levels → CurriculumScore ranking |
| `src/render_surfing.py` | Render the graph in the Poincaré disk and half-plane (SVG with data-* attributes for interactivity) |
| `src/hyperbolic_engine.py` | Low-level geometry: Möbius transforms, geodesic arcs, radial PPR mapping |
| `data/poincare_art_data_*.json` | Generated graph data (nodes, edges, PPR, coordinates) |
| `assets/*.svg` | Rendered visualizations |
| `docs/index.html` | Interactive GitHub Pages demo |
| `theory/MATHEMATICAL_FOUNDATIONS.md` | Full derivation (Sections 9-20: SCC, bandits, entropy, HMM, MDP, SARSA, Dyna, counterfactuals, hyperbolic geometry) |
| `theory/From_Cycles_to_Curvature.docx` | Plain-language walkthrough of the full pipeline |

## Pipeline

1. **Graph construction** — nodes are concepts, directed edges are "refers to" relations. Domains partition nodes into angular sectors.
2. **Personalized PageRank** — 50,000 random walks with α=0.15 restart probability give each node a stationary importance.
3. **Tarjan's SCC (iterative)** — finds mutually-referencing concept clusters. For surfing: 77 SCCs, one giant SCC of 63 nodes (the interconnected core of surfing culture/practice/oceanography).
4. **Condensation DAG** — collapse each SCC to a super-node. 47% compression on the surfing graph.
5. **Topological level assignment** — BFS from sources (in-degree 0). Level ℓ(v) = length of longest path from any source.
6. **CurriculumScore** — rank within each level by PPR(v) / (1 + ℓ(v)). High-PPR basis concepts come first.
7. **Hyperbolic layout** — radial coordinate = f(PPR) (central concepts near the origin), angular coordinate = domain sector + within-domain rank. Geodesics rendered as circular arcs orthogonal to the boundary.

## Results (surfing, depth ≤ 3)

- **146 nodes, 236 directed edges, 10 domains**
- **77 SCCs** (70 singletons, 7 multi-node). Largest SCC: 63 nodes spanning surfing culture, oceanography, competition, and environment — these concepts define each other mutually.
- **Max depth: 5.** 96% of total PPR mass sits at depth ≤ 3.
- **Level 0 (basis, no prerequisites):** Jeffreys Bay, Tropical cyclone, Tide, Surf photography, ...
- **42 gateway nodes** (articulation points) — removing any one fragments the graph.

## Run locally

```bash
pip install -r requirements.txt          # numpy, (no other hard deps)
python src/generate_surfing_data.py      # → data/poincare_art_data_surfing.json
python src/scc_surfing.py                # prints full curriculum
python src/render_surfing.py             # → assets/surfing_{disk,halfplane}.svg
```

Then open `docs/index.html` locally, or push to `gh-pages` and serve from `/docs`.

## Why hyperbolic?

Knowledge graphs have **exponentially growing neighborhoods** (each concept branches into many sub-concepts, which branch further). Hyperbolic space also has exponential area growth — so a tree of concepts embeds with low distortion, where Euclidean layouts crowd the periphery. The Poincaré disk puts high-PPR "basis" concepts at the center and pushes specialized concepts to the boundary where there's room to see detail.

## License

MIT.
