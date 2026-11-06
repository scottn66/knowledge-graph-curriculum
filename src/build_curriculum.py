#!/usr/bin/env python3
"""Build a reading curriculum from any English Wikipedia seed.

Defaults match the Optics proof of concept: depth 2, fanout 15, 150 articles,
9,000 random walks of length 20, restart probability 0.15.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kgc.pipeline import run_pipeline  # noqa: E402
from kgc.wiki import WikiError  # noqa: E402
import urllib.error  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, help="English Wikipedia article title")
    parser.add_argument("--depth", type=int, default=2, help="BFS expansion depth (default 2)")
    parser.add_argument("--fanout", type=int, default=15, help="prose outlinks kept per article")
    parser.add_argument("--max-nodes", type=int, default=150, help="article cap")
    parser.add_argument("--walks", type=int, default=9000, help="personalized PageRank walks")
    parser.add_argument("--walk-len", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.15, help="restart probability")
    parser.add_argument(
        "--sparsify-k",
        type=int,
        default=0,
        help="if > 0, keep the k outlinks with the highest target PPR before SCC",
    )
    parser.add_argument("--tau", type=float, default=1.0, help="hyperbolic tree edge length")
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--vault",
        default=None,
        help="Obsidian vault directory (default: vaults/<seed>-vault)",
    )
    parser.add_argument("--web-json", default=None, help="also write the curriculum JSON here")
    parser.add_argument("--curriculum-txt", default=None, help="also write a plain-text curriculum")
    parser.add_argument("--pause", type=float, default=0.05, help="seconds between API calls")
    args = parser.parse_args(argv)

    vault = args.vault
    if vault is None:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in args.seed).strip("-")
        slug = "-".join(part for part in slug.split("-") if part) or "seed"
        vault = str(ROOT / "vaults" / f"{slug}-vault")

    from kgc.wiki import MediaWiki

    def progress(message: str, done: int, total: int) -> None:
        print(f"[{done}/{total}] {message}", file=sys.stderr)

    try:
        document = run_pipeline(
            args.seed,
            depth=args.depth,
            fanout=args.fanout,
            max_nodes=args.max_nodes,
            n_walks=args.walks,
            walk_len=args.walk_len,
            alpha=args.alpha,
            sparsify_k=args.sparsify_k,
            tau=args.tau,
            rng_seed=args.rng_seed,
            vault=vault,
            web_json=args.web_json,
            curriculum_txt=args.curriculum_txt,
            wiki=MediaWiki(pause=args.pause),
            on_progress=progress,
        )
    except (WikiError, urllib.error.URLError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stats = document["stats"]
    print(
        f"{document['seed']}: {stats['nodes']} articles, {stats['edges']} links, "
        f"{stats['scc_count']} SCCs ({stats['multi_sccs']} multi, largest {stats['largest_scc']}), "
        f"depth {stats['max_level']}"
    )
    print(f"vault: {vault}")
    counts = stats["level_counts"]
    for level in sorted(counts, key=int):
        print(f"  level {level}: {counts[level]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
