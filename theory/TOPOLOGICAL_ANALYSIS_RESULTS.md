# Topological Analysis Results: ref_counts.db

## Dataset
- **Pages**: 193,889 total (1,387 with content, 192,502 frontier stubs)
- **Edges**: 562,711 total (45,635 within crawled subgraph)
- **Crawl seed**: Artificial_intelligence, BFS depth 2-3, max_connections=8

## The Giant-SCC Problem

### Raw analysis (no filtering)
All 1,387 crawled pages form a **single strongly connected component**. This means every page can reach every other page through hyperlinks. Topological depth: 0 (everything at the same level).

**Why**: BFS crawls capture the densely interconnected core of Wikipedia. With avg out-degree ~33, every pair of crawled pages connects through short paths. Structural pages (Main_Page with 1,387 in-links, ISBN with 1,041, Doi with 875) create artificial full-connectivity.

### Metadata filtering alone
Removing 35 metadata/identifier pages still leaves a giant SCC of 1,289 pages. The content pages themselves are too densely linked.

## Solution: PPR + Edge Sparsification

### Algorithm
1. **Personalized PageRank** from seed "Cognitive_science" (alpha=0.15, 67 iterations to convergence)
2. **Top-k subgraph**: Take 150 highest-PPR content pages
3. **k-NN edge pruning**: Each node keeps only its top-3 outgoing edges (ranked by target PPR score)
4. **SCC condensation** via Kosaraju's algorithm on the sparsified graph
5. **Topological sort** of condensation DAG via Kahn's algorithm

### Results (k=3, top-150)
- **Edges**: 409 (down from 3,508 in the top-200 dense subgraph)
- **SCCs**: 136 total, 8 non-trivial, largest has 5 members
- **Topological depth**: 7 levels (meaningful hierarchy!)

### Sparsification comparison
| k | Edges | SCCs | Largest SCC | Max Depth |
|---|-------|------|-------------|-----------|
| 3 | 568   | 177  | 10          | 9         |
| 5 | 904   | 131  | 66          | 7         |
| 8 | 1331  | 71   | 129         | 4         |

**k=3** is the sweet spot: enough edges for meaningful topology, sparse enough to break cycles.

## Discovered Concept Clusters (SCCs)

These are sets of articles that mutually reference each other at the same conceptual level:

1. **Core cluster** (Level 7, 5 articles): Cognitive Science, Artificial Intelligence, Aristotle, Plato, Noam Chomsky
2. **Statistics-Philosophy** (Level 5, 2 articles): Statistics, Philosophy of Mathematics
3. **VR/AR cluster** (Level 5, 4 articles): Augmented Reality, Meta Platforms, Instagram, Oculus Rift
4. **Bayesian cluster** (Level 1, 2 articles): Bayesian Probability, Bayes Factor

## Prerequisite Hierarchy

### Level 0 (99 articles) — No prerequisites, start here
Specific sub-topics: Kin Recognition, Neo-Darwinism, Nature vs Nurture, Evolutionary Linguistics, Dual Process Theory, Number Sense, Mental State, Steven Pinker, Jerry Fodor, Philosophy of AI, Rene Descartes, Knowledge Representation, Deep Learning, Turing Machine, Linear Algebra, Time Series, ...

### Level 1 (18 articles) — Requires Level 0
History of Evolutionary Thought, Daniel Dennett, Socialization, Gottfried Leibniz, Allen Newell, Mathematical Logic, Isaac Newton, Bayesian Probability cluster, ...

### Level 2 (9 articles) — Requires Levels 0-1
Behaviorism, Punctuated Equilibrium, Automata Theory, John Locke, Bayesian Linear Regression, ...

### Level 3 (5 articles) — Requires Levels 0-2
David Hume, Bertrand Russell, Geographic Information System, ...

### Level 4 (3 articles) — Requires Levels 0-3
Data Mining, Fourier Analysis

### Level 5 (7 articles) — Requires Levels 0-4
Statistics + Philosophy of Mathematics (co-referential cluster), Augmented Reality cluster, Physics

### Level 6 (4 articles) — Requires Levels 0-5
Mathematics, Encyclopedia reference cluster

### Level 7 (5 articles) — Requires all levels
**Cognitive Science, Artificial Intelligence, Aristotle, Plato, Noam Chomsky** — the apex concept cluster

## Interpretation

The hierarchy reveals that in this crawl, **Cognitive Science sits at the apex** of a 7-level prerequisite structure. The algorithm correctly identifies:

- **Narrow entry points** (Level 0): Topics specific enough to be understood independently
- **Integrative mid-level concepts** (Levels 1-4): Figures and theories that bridge domains
- **Foundational pillars** (Levels 5-6): Mathematics, Statistics, Philosophy of Mathematics
- **Central hub** (Level 7): The seed topic and its most tightly coupled concepts

The inversion (specific topics at bottom, general at top) reflects how Wikipedia's hyperlink structure works: specialized articles reference general ones, creating a "points up" topology where generality increases with topological depth.

## CurriculumScore Formula

For each article v with seed s:

    CurriculumScore(v) = PPR(v; s, alpha) / (1 + level(v))

This balances topic relevance (PPR) with prerequisite depth (level). Articles with high PPR and low level are the best starting points for a newcomer to the seed topic.

## Technical Notes

- Kosaraju's algorithm (iterative, O(V+E)) used instead of recursive Tarjan's to avoid Python stack overflow on 193K nodes
- PPR converges in 67 iterations with tol=1e-8
- k=3 sparsification is equivalent to keeping only the "3 most important references" per article
- The giant-SCC phenomenon is expected for BFS crawls — it validates the need for sparsification in any topological analysis of web graphs
