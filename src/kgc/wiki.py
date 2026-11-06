"""Constrained Wikipedia outlink extraction.

Links are taken from article prose in reading order. Infoboxes, citations,
and other templates are removed first so the fanout is the lead and body,
not the chrome that ties every page into one giant component.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "knowledge-graph-curriculum/1.0 "
    "(https://github.com/scottn66/knowledge-graph-curriculum; curriculum research)"
)

# Keep in sync with docs/curriculum.js NAMESPACES / META_TITLES.
NAMESPACES = frozenset({
    "file", "image", "media", "category", "special", "wikipedia", "wp",
    "project", "help", "template", "portal", "draft", "module", "timedtext",
    "book", "mediawiki", "user", "talk", "user talk", "wikipedia talk",
    "file talk", "image talk", "template talk", "help talk", "category talk",
    "portal talk", "draft talk", "module talk", "mediawiki talk", "book talk",
    "wikt", "wiktionary", "commons", "c", "q", "d", "v", "n", "b", "s", "w",
    "m", "voy", "meta", "phabricator", "mw",
})

META_TITLES = frozenset({
    "main page",
    "isbn",
    "doi",
    "issn",
    "pmid",
    "pmc (identifier)",
    "s2cid",
    "bibcode",
    "jstor",
    "oclc",
    "digital object identifier",
    "international standard book number",
    "international standard serial number",
    "pubmed",
    "pubmed central",
    "semantic scholar",
    "wayback machine",
    "wikidata",
    "wikimedia commons",
})


def _find_close(text: str, start: int, open_tok: str, close_tok: str) -> int | None:
    depth = 0
    i = start
    n = len(text)
    ol = len(open_tok)
    cl = len(close_tok)
    while i < n:
        if text.startswith(open_tok, i):
            depth += 1
            i += ol
            continue
        if text.startswith(close_tok, i):
            depth -= 1
            i += cl
            if depth == 0:
                return i
            continue
        i += 1
    return None


def strip_comments(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("<!--", i):
            close = text.find("-->", i + 4)
            if close == -1:
                break
            i = close + 3
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def strip_refs(text: str) -> str:
    low = text.lower()
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if low.startswith("<ref", i) and (i + 4 == n or not low[i + 4].isalpha()):
            gt = low.find(">", i)
            if gt == -1:
                out.append(text[i])
                i += 1
                continue
            if low[gt - 1] == "/":
                i = gt + 1
                continue
            close = low.find("</ref>", gt)
            if close == -1:
                out.append(text[i])
                i += 1
                continue
            i = close + len("</ref>")
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def strip_templates(text: str) -> str:
    """Drop balanced {{...}} regions. If braces are unbalanced, keep the text."""
    depth = 0
    saw_open = False
    i = 0
    n = len(text)
    kept: list[str] = []
    while i < n:
        if text.startswith("{{", i):
            depth += 1
            saw_open = True
            i += 2
            continue
        if depth and text.startswith("}}", i):
            depth -= 1
            i += 2
            continue
        if depth == 0:
            kept.append(text[i])
        i += 1
    if depth != 0:
        return text
    return "".join(kept) if saw_open else text


def prose_wikitext(text: str) -> str:
    return strip_templates(strip_refs(strip_comments(text)))


def _link_target(inner: str) -> str | None:
    depth = 0
    buf: list[str] = []
    i = 0
    while i < len(inner):
        if inner.startswith("[[", i):
            depth += 1
            i += 2
            continue
        if inner.startswith("]]", i) and depth:
            depth -= 1
            i += 2
            continue
        if depth == 0 and inner[i] in "|#":
            break
        buf.append(inner[i])
        i += 1
    title = re.sub(r"\s+", " ", "".join(buf).replace("_", " ")).strip()
    if not title or title.startswith("#") or title.startswith(":"):
        return None
    if ":" in title:
        prefix = title.split(":", 1)[0].strip().lower()
        if prefix in NAMESPACES or re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})?", prefix):
            return None
    if title.casefold() in META_TITLES:
        return None
    if title.casefold().endswith("(disambiguation)"):
        return None
    return title


def extract_prose_links(wikitext: str, fanout: int | None = None) -> list[str]:
    """Unique mainspace links in prose order. `fanout` caps the list."""
    text = prose_wikitext(wikitext or "")
    found: list[str] = []
    seen: set[str] = set()
    i = 0
    n = len(text)
    while i < n - 1:
        if text.startswith("[[", i):
            close = _find_close(text, i, "[[", "]]")
            if close is None:
                i += 2
                continue
            target = _link_target(text[i + 2 : close - 2])
            i = close
            if not target:
                continue
            key = target.casefold()
            if key in seen:
                continue
            seen.add(key)
            found.append(target)
            if fanout is not None and len(found) >= fanout:
                break
            continue
        i += 1
    return found


def wiki_url(title: str) -> str:
    slug = title.replace(" ", "_")
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(slug, safe="()_!'*,-.")


@dataclass
class CrawlResult:
    seed: str
    requested_seed: str
    depth_of: dict[str, int]
    parent: dict[str, str | None]
    edges: list[tuple[str, str]]
    params: dict = field(default_factory=dict)


class WikiError(RuntimeError):
    pass


class MediaWiki:
    def __init__(self, pause: float = 0.05):
        self.pause = pause

    def _get(self, params: dict) -> dict:
        query = urllib.parse.urlencode({
            "format": "json",
            "formatversion": "2",
            "origin": "*",
            "maxlag": "5",
            **params,
        })
        request = urllib.request.Request(
            f"{API}?{query}",
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        last_error: Exception | None = None
        for attempt in range(4):
            if attempt:
                time.sleep(min(8, 0.5 * 2**attempt))
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in {429, 502, 503, 504}:
                    retry = exc.headers.get("Retry-After") if exc.headers else None
                    if retry and str(retry).isdigit():
                        time.sleep(min(30, int(retry)))
                    continue
                raise WikiError(f"Wikipedia HTTP {exc.code}") from exc
            except urllib.error.URLError as exc:
                last_error = exc
                continue
            error = payload.get("error")
            if error:
                code = error.get("code")
                if code == "maxlag":
                    last_error = WikiError(error.get("info", "maxlag"))
                    continue
                raise WikiError(error.get("info") or str(error))
            if self.pause:
                time.sleep(self.pause)
            return payload
        raise WikiError(f"Wikipedia request failed: {last_error}")

    def _pages(self, titles: list[str], content: bool) -> dict:
        if not titles:
            return {"pages": [], "redirects": [], "normalized": []}
        params = {
            "action": "query",
            "redirects": "1",
            "titles": "|".join(titles),
        }
        if content:
            params.update({
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
            })
        merged = {"pages": [], "redirects": [], "normalized": []}
        # Manual continuation is unnecessary for a single revision, but a huge
        # batch can still come back incomplete.
        while True:
            payload = self._get(params)
            query = payload.get("query") or {}
            merged["pages"].extend(query.get("pages") or [])
            merged["redirects"].extend(query.get("redirects") or [])
            merged["normalized"].extend(query.get("normalized") or [])
            cont = payload.get("continue")
            if not cont:
                return merged
            params = {**params, **cont}

    def resolve(self, titles: list[str]) -> dict[str, str | None]:
        """Map each requested title to its canonical title, or None if missing."""
        out: dict[str, str | None] = {title: None for title in titles}
        for start in range(0, len(titles), 40):
            batch = titles[start : start + 40]
            data = self._pages(batch, content=False)
            alias: dict[str, str] = {}
            for row in data["normalized"]:
                alias[row["from"]] = row["to"]
            for row in data["redirects"]:
                alias[row["from"]] = row["to"]
            live = {}
            for page in data["pages"]:
                if page.get("missing") or page.get("invalid"):
                    continue
                live[page["title"]] = page["title"]

            def canon(title: str) -> str | None:
                seen: set[str] = set()
                current = title
                while current in alias and current not in seen:
                    seen.add(current)
                    current = alias[current]
                if current in live:
                    return live[current]
                # Normalization can change the key before the page table.
                folded = {name.casefold(): name for name in live}
                return folded.get(current.casefold())

            for title in batch:
                out[title] = canon(title)
        return out

    def wikitext(self, titles: list[str]) -> dict[str, str]:
        """Map canonical title → wikitext. Redirects are followed."""
        found: dict[str, str] = {}
        for start in range(0, len(titles), 8):
            batch = titles[start : start + 8]
            data = self._pages(batch, content=True)
            for page in data["pages"]:
                if page.get("missing") or page.get("invalid"):
                    continue
                revisions = page.get("revisions") or []
                if not revisions:
                    continue
                slots = (revisions[0].get("slots") or {}).get("main") or {}
                content = slots.get("content")
                if content is None:
                    content = revisions[0].get("*") or revisions[0].get("content") or ""
                found[page["title"]] = content
        return found


def crawl(
    seed: str,
    *,
    depth: int = 2,
    fanout: int = 15,
    max_nodes: int = 150,
    wiki: MediaWiki | None = None,
    on_progress=None,
) -> CrawlResult:
    """BFS outlink crawl. `depth` is the longest path of expansions from the seed."""
    if depth < 0:
        raise ValueError("depth must be >= 0")
    if fanout < 1:
        raise ValueError("fanout must be >= 1")
    if max_nodes < 1:
        raise ValueError("max_nodes must be >= 1")
    client = wiki or MediaWiki()
    requested = seed.strip()
    if not requested:
        raise WikiError("Seed title is empty")

    resolved = client.resolve([requested])
    canonical = resolved.get(requested)
    if not canonical:
        raise WikiError(f"No English Wikipedia article for {requested!r}")

    depth_of = {canonical: 0}
    parent: dict[str, str | None] = {canonical: None}
    edges: list[tuple[str, str]] = []
    edge_seen: set[tuple[str, str]] = set()
    expand: deque[str] = deque([canonical])

    def report(message: str) -> None:
        if on_progress:
            on_progress(message, len(depth_of), max_nodes)

    report(f"Seed resolved to {canonical}")
    while expand and len(depth_of) < max_nodes:
        batch: list[str] = []
        while expand and len(batch) < 8:
            title = expand.popleft()
            if depth_of[title] >= depth:
                continue
            batch.append(title)
        if not batch:
            break
        report(f"Reading {batch[0]}" + (f" and {len(batch) - 1} more" if len(batch) > 1 else ""))
        texts = client.wikitext(batch)
        extracted: dict[str, list[str]] = {}
        to_resolve: list[str] = []
        for title in batch:
            links = extract_prose_links(texts.get(title, ""), fanout=max(fanout * 3, fanout))
            extracted[title] = links
            to_resolve.extend(links)
        # Preserve order while dropping duplicates for the resolve call.
        uniq_resolve: list[str] = []
        seen_resolve: set[str] = set()
        for title in to_resolve:
            key = title.casefold()
            if key in seen_resolve:
                continue
            seen_resolve.add(key)
            uniq_resolve.append(title)
        canon_of = client.resolve(uniq_resolve) if uniq_resolve else {}
        for title in batch:
            kept: list[str] = []
            kept_keys: set[str] = set()
            for link in extracted[title]:
                target = canon_of.get(link)
                if not target or target == title:
                    continue
                key = target.casefold()
                if key in kept_keys:
                    continue
                kept_keys.add(key)
                kept.append(target)
                if len(kept) >= fanout:
                    break
            for target in kept:
                pair = (title, target)
                if pair not in edge_seen:
                    edge_seen.add(pair)
                    edges.append(pair)
                if target not in depth_of:
                    if len(depth_of) >= max_nodes:
                        break
                    depth_of[target] = depth_of[title] + 1
                    parent[target] = title
                    expand.append(target)
        report(f"{len(depth_of)} articles in the graph")

    return CrawlResult(
        seed=canonical,
        requested_seed=requested,
        depth_of=depth_of,
        parent=parent,
        edges=edges,
        params={"depth": depth, "fanout": fanout, "max_nodes": max_nodes},
    )
