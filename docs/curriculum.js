/* Knowledge-graph curriculum in the browser.
   Port of src/kgc: prose outlinks, iterative Tarjan, condensation levels,
   random-walk PPR, CurriculumScore, Sarkar disk embedding.
   Keep the namespace list and the scanners aligned with src/kgc/wiki.py. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.KGC = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const NAMESPACES = new Set([
    "file", "image", "media", "category", "special", "wikipedia", "wp",
    "project", "help", "template", "portal", "draft", "module", "timedtext",
    "book", "mediawiki", "user", "talk", "user talk", "wikipedia talk",
    "file talk", "image talk", "template talk", "help talk", "category talk",
    "portal talk", "draft talk", "module talk", "mediawiki talk", "book talk",
    "wikt", "wiktionary", "commons", "c", "q", "d", "v", "n", "b", "s", "w",
    "m", "voy", "meta", "phabricator", "mw",
  ]);

  const META = new Set([
    "main page", "isbn", "doi", "issn", "pmid", "pmc (identifier)", "s2cid",
    "bibcode", "jstor", "oclc", "digital object identifier",
    "international standard book number", "international standard serial number",
    "pubmed", "pubmed central", "semantic scholar", "wayback machine",
    "wikidata", "wikimedia commons",
  ]);

  const API = "https://en.wikipedia.org/w/api.php";

  function findClose(text, start, openTok, closeTok) {
    let depth = 0;
    let i = start;
    const n = text.length;
    while (i < n) {
      if (text.startsWith(openTok, i)) {
        depth += 1;
        i += openTok.length;
        continue;
      }
      if (text.startsWith(closeTok, i)) {
        depth -= 1;
        i += closeTok.length;
        if (depth === 0) return i;
        continue;
      }
      i += 1;
    }
    return null;
  }

  function stripComments(text) {
    let out = "";
    let i = 0;
    while (i < text.length) {
      if (text.startsWith("<!--", i)) {
        const close = text.indexOf("-->", i + 4);
        if (close === -1) break;
        i = close + 3;
        continue;
      }
      out += text[i];
      i += 1;
    }
    return out;
  }

  function stripRefs(text) {
    const low = text.toLowerCase();
    let out = "";
    let i = 0;
    while (i < text.length) {
      if (low.startsWith("<ref", i) && (i + 4 === low.length || !/[a-z]/.test(low[i + 4]))) {
        const gt = low.indexOf(">", i);
        if (gt === -1) {
          out += text[i];
          i += 1;
          continue;
        }
        if (low[gt - 1] === "/") {
          i = gt + 1;
          continue;
        }
        const close = low.indexOf("</ref>", gt);
        if (close === -1) {
          out += text[i];
          i += 1;
          continue;
        }
        i = close + "</ref>".length;
        continue;
      }
      out += text[i];
      i += 1;
    }
    return out;
  }

  function stripTemplates(text) {
    let depth = 0;
    let saw = false;
    let kept = "";
    let i = 0;
    while (i < text.length) {
      if (text.startsWith("{{", i)) {
        depth += 1;
        saw = true;
        i += 2;
        continue;
      }
      if (depth && text.startsWith("}}", i)) {
        depth -= 1;
        i += 2;
        continue;
      }
      if (depth === 0) kept += text[i];
      i += 1;
    }
    if (depth !== 0) return text;
    return saw ? kept : text;
  }

  function linkTarget(inner) {
    let depth = 0;
    let buf = "";
    let i = 0;
    while (i < inner.length) {
      if (inner.startsWith("[[", i)) {
        depth += 1;
        i += 2;
        continue;
      }
      if (inner.startsWith("]]", i) && depth) {
        depth -= 1;
        i += 2;
        continue;
      }
      if (depth === 0 && (inner[i] === "|" || inner[i] === "#")) break;
      buf += inner[i];
      i += 1;
    }
    const title = buf.replaceAll("_", " ").replace(/\s+/g, " ").trim();
    if (!title || title.startsWith("#") || title.startsWith(":")) return null;
    if (title.includes(":")) {
      const prefix = title.split(":", 1)[0].trim().toLowerCase();
      if (NAMESPACES.has(prefix) || /^[a-z]{2,3}(?:-[a-z0-9]{2,8})?$/.test(prefix)) return null;
    }
    if (META.has(title.toLowerCase())) return null;
    if (title.toLowerCase().endsWith("(disambiguation)")) return null;
    return title;
  }

  function extractProseLinks(wikitext, fanout) {
    const text = stripTemplates(stripRefs(stripComments(wikitext || "")));
    const found = [];
    const seen = new Set();
    let i = 0;
    while (i < text.length - 1) {
      if (text.startsWith("[[", i)) {
        const close = findClose(text, i, "[[", "]]");
        if (close === null) {
          i += 2;
          continue;
        }
        const target = linkTarget(text.slice(i + 2, close - 2));
        i = close;
        if (!target) continue;
        const key = target.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        found.push(target);
        if (fanout != null && found.length >= fanout) break;
        continue;
      }
      i += 1;
    }
    return found;
  }

  function wikiUrl(title) {
    const slug = title.replaceAll(" ", "_");
    return "https://en.wikipedia.org/wiki/" + encodeURIComponent(slug)
      .replace(/%2C/g, ",")
      .replace(/%21/g, "!")
      .replace(/%28/g, "(")
      .replace(/%29/g, ")")
      .replace(/%2A/g, "*")
      .replace(/%27/g, "'");
  }

  function iterativeTarjan(nodes, adj) {
    const index = new Map();
    const lowlink = new Map();
    const stack = [];
    const onStack = new Set();
    const sccs = [];
    let counter = 0;

    function strongconnect(start) {
      index.set(start, counter);
      lowlink.set(start, counter);
      counter += 1;
      stack.push(start);
      onStack.add(start);
      const work = [[start, 0]];
      while (work.length) {
        const top = work[work.length - 1];
        const node = top[0];
        const ni = top[1];
        const neighbors = adj.get(node) || [];
        if (ni < neighbors.length) {
          top[1] = ni + 1;
          const next = neighbors[ni];
          if (!index.has(next)) {
            index.set(next, counter);
            lowlink.set(next, counter);
            counter += 1;
            stack.push(next);
            onStack.add(next);
            work.push([next, 0]);
          } else if (onStack.has(next)) {
            lowlink.set(node, Math.min(lowlink.get(node), index.get(next)));
          }
        } else {
          if (lowlink.get(node) === index.get(node)) {
            const component = [];
            while (true) {
              const popped = stack.pop();
              onStack.delete(popped);
              component.push(popped);
              if (popped === node) break;
            }
            sccs.push(component);
          }
          work.pop();
          if (work.length) {
            const parent = work[work.length - 1][0];
            lowlink.set(parent, Math.min(lowlink.get(parent), lowlink.get(node)));
          }
        }
      }
    }

    for (const node of nodes) {
      if (!index.has(node)) strongconnect(node);
    }
    return sccs;
  }

  function preparedAdjacency(nodes, edges) {
    const present = new Set(nodes);
    const raw = new Map(nodes.map((node) => [node, new Set()]));
    for (const [src, dst] of edges) {
      if (present.has(src) && present.has(dst) && src !== dst) raw.get(src).add(dst);
    }
    const adj = new Map();
    for (const node of nodes) adj.set(node, [...raw.get(node)].sort());
    return adj;
  }

  function orderSccs(sccs) {
    return [...sccs].sort((a, b) => {
      if (b.length !== a.length) return b.length - a.length;
      const amin = [...a].sort()[0];
      const bmin = [...b].sort()[0];
      if (amin < bmin) return -1;
      if (amin > bmin) return 1;
      return 0;
    });
  }

  function condense(nodes, edges, sccs) {
    const membership = new Map();
    sccs.forEach((component, id) => {
      for (const node of component) membership.set(node, id);
    });
    const dag = new Set();
    for (const [src, dst] of edges) {
      if (!membership.has(src) || !membership.has(dst)) continue;
      const a = membership.get(src);
      const b = membership.get(dst);
      if (a !== b) dag.add(a + "\0" + b);
    }
    const dagEdges = [...dag].map((key) => key.split("\0").map(Number));
    dagEdges.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    return { membership, dagEdges };
  }

  function longestPathLevels(sccCount, dagEdges) {
    const outgoing = new Map();
    const indeg = Array(sccCount).fill(0);
    for (const [src, dst] of dagEdges) {
      if (!outgoing.has(src)) outgoing.set(src, []);
      outgoing.get(src).push(dst);
      indeg[dst] += 1;
    }
    const queue = [];
    for (let i = 0; i < sccCount; i += 1) if (indeg[i] === 0) queue.push(i);
    const work = indeg.slice();
    const order = [];
    let head = 0;
    while (head < queue.length) {
      const node = queue[head];
      head += 1;
      order.push(node);
      for (const next of outgoing.get(node) || []) {
        work[next] -= 1;
        if (work[next] === 0) queue.push(next);
      }
    }
    if (order.length !== sccCount) throw new Error("condensation has a cycle");
    const level = Array(sccCount).fill(0);
    for (const node of order) {
      for (const next of outgoing.get(node) || []) {
        level[next] = Math.max(level[next], level[node] + 1);
      }
    }
    return level;
  }

  function analyze(nodes, edges) {
    const titles = [...nodes].sort();
    const adj = preparedAdjacency(titles, edges);
    const sccs = orderSccs(iterativeTarjan(titles, adj));
    const { membership, dagEdges } = condense(titles, edges, sccs);
    const levels = longestPathLevels(sccs.length, dagEdges);
    const byTitle = {};
    for (const title of titles) {
      const id = membership.get(title);
      byTitle[title] = { sccId: id, level: levels[id], sccSize: sccs[id].length };
    }
    return { sccs, dagEdges, byTitle };
  }

  function curriculumScore(ppr, level) {
    return ppr / (1 + level);
  }

  function randomWalkPpr(nodes, adj, seed, opts) {
    const nWalks = opts.nWalks;
    const walkLen = opts.walkLen;
    const alpha = opts.alpha;
    const random = opts.random || Math.random;
    const counts = new Map(nodes.map((node) => [node, 0]));
    for (let walk = 0; walk < nWalks; walk += 1) {
      let current = seed;
      counts.set(current, counts.get(current) + 1);
      for (let step = 0; step < walkLen; step += 1) {
        const neighbors = adj.get(current) || [];
        if (!neighbors.length || random() < alpha) current = seed;
        else current = neighbors[Math.floor(random() * neighbors.length)];
        counts.set(current, counts.get(current) + 1);
      }
    }
    const total = [...counts.values()].reduce((sum, value) => sum + value, 0);
    const out = new Map();
    for (const node of nodes) out.set(node, counts.get(node) / total);
    return out;
  }

  function articulationPoints(nodes, edges) {
    const present = new Set(nodes);
    const undirected = new Map(nodes.map((node) => [node, new Set()]));
    for (const [src, dst] of edges) {
      if (present.has(src) && present.has(dst) && src !== dst) {
        undirected.get(src).add(dst);
        undirected.get(dst).add(src);
      }
    }
    const adj = new Map();
    for (const node of nodes) adj.set(node, [...undirected.get(node)].sort());
    const discovery = new Map();
    const low = new Map();
    const parent = new Map();
    const points = new Set();
    let timer = 0;
    for (const start of nodes) {
      if (discovery.has(start)) continue;
      parent.set(start, null);
      discovery.set(start, timer);
      low.set(start, timer);
      timer += 1;
      let rootChildren = 0;
      const stack = [[start, 0]];
      while (stack.length) {
        const top = stack[stack.length - 1];
        const node = top[0];
        const ni = top[1];
        const neighbors = adj.get(node);
        if (ni < neighbors.length) {
          top[1] = ni + 1;
          const next = neighbors[ni];
          if (!discovery.has(next)) {
            parent.set(next, node);
            discovery.set(next, timer);
            low.set(next, timer);
            timer += 1;
            if (node === start) rootChildren += 1;
            stack.push([next, 0]);
          } else if (next !== parent.get(node)) {
            low.set(node, Math.min(low.get(node), discovery.get(next)));
          }
        } else {
          stack.pop();
          const par = parent.get(node);
          if (par != null) {
            low.set(par, Math.min(low.get(par), low.get(node)));
            if (parent.get(par) != null && low.get(node) >= discovery.get(par)) points.add(par);
          }
        }
      }
      if (rootChildren > 1) points.add(start);
    }
    return [...points].sort();
  }

  function cadd(z, w) {
    const cwRe = z.re * w.re + z.im * w.im;
    const cwIm = z.re * w.im - z.im * w.re;
    const dRe = 1 + cwRe;
    const dIm = cwIm;
    const nRe = z.re + w.re;
    const nIm = z.im + w.im;
    const den = dRe * dRe + dIm * dIm;
    if (den < 1e-30) return { re: 0, im: 0 };
    return { re: (nRe * dRe + nIm * dIm) / den, im: (nIm * dRe - nRe * dIm) / den };
  }

  function csub(z, w) {
    return { re: z.re - w.re, im: z.im - w.im };
  }

  function cdiv(z, w) {
    const den = w.re * w.re + w.im * w.im;
    return { re: (z.re * w.re + z.im * w.im) / den, im: (z.im * w.re - z.re * w.im) / den };
  }

  function towardParent(parent, local) {
    const child = cadd(parent, local);
    const mag = Math.hypot(child.re, child.im);
    if (mag < 1e-12) return Math.atan2(-local.im, -local.re);
    const diff = csub(parent, child);
    const conjChild = { re: child.re, im: -child.im };
    const prod = {
      re: conjChild.re * parent.re - conjChild.im * parent.im,
      im: conjChild.re * parent.im + conjChild.im * parent.re,
    };
    const denom = { re: 1 - prod.re, im: -prod.im };
    if (Math.hypot(denom.re, denom.im) < 1e-15) return Math.atan2(-local.im, -local.re);
    const mapped = cdiv(diff, denom);
    return Math.atan2(mapped.im, mapped.re);
  }

  function sarkarEmbed(root, children, tau) {
    const sizes = new Map();
    function measure(node) {
      let total = 1;
      for (const child of children.get(node) || []) total += measure(child);
      sizes.set(node, total);
      return total;
    }
    measure(root);
    const rho = Math.tanh(tau / 2);
    const positions = new Map([[root, { re: 0, im: 0 }]]);
    function place(node, incoming) {
      const kids = children.get(node) || [];
      if (!kids.length) return;
      let coneLeft;
      let coneWidth;
      if (incoming == null) {
        coneLeft = -Math.PI;
        coneWidth = 2 * Math.PI;
      } else {
        coneWidth = Math.PI;
        coneLeft = incoming + Math.PI - coneWidth / 2;
      }
      const weight = kids.reduce((sum, child) => sum + sizes.get(child), 0);
      let cursor = coneLeft;
      for (const child of kids) {
        const span = coneWidth * (sizes.get(child) / weight);
        const mid = cursor + span / 2;
        const local = { re: rho * Math.cos(mid), im: rho * Math.sin(mid) };
        positions.set(child, cadd(positions.get(node), local));
        place(child, towardParent(positions.get(node), local));
        cursor += span;
      }
    }
    place(root, null);
    return positions;
  }

  function childrenFromParents(root, parent) {
    const children = new Map();
    for (const node of Object.keys(parent)) children.set(node, []);
    for (const [node, par] of Object.entries(parent)) {
      if (node === root || par == null) continue;
      if (!children.has(par)) children.set(par, []);
      children.get(par).push(node);
    }
    for (const [node, kids] of children) children.set(node, kids.sort());
    return children;
  }

  function round(value, digits) {
    const scale = 10 ** digits;
    return Math.round(value * scale) / scale;
  }

  function assembleCurriculum(crawled, opts) {
    const options = Object.assign({
      nWalks: 9000,
      walkLen: 20,
      alpha: 0.15,
      tau: 1.0,
      random: Math.random,
    }, opts || {});
    const titles = Object.keys(crawled.depthOf).sort();
    const edges = crawled.edges.map((edge) => [edge[0], edge[1]]);
    const adj = preparedAdjacency(titles, edges);
    const sccs = orderSccs(iterativeTarjan(titles, adj));
    const { membership, dagEdges } = condense(titles, edges, sccs);
    const sccLevels = longestPathLevels(sccs.length, dagEdges);
    const ppr = randomWalkPpr(titles, adj, crawled.seed, options);
    const gateways = new Set(articulationPoints(titles, edges));
    const children = childrenFromParents(crawled.seed, crawled.parent);
    const positions = sarkarEmbed(crawled.seed, children, options.tau);
    const outDegree = new Map(titles.map((title) => [title, 0]));
    const inDegree = new Map(titles.map((title) => [title, 0]));
    const tree = new Set();
    for (const [title, par] of Object.entries(crawled.parent)) {
      if (par) tree.add(par + "\0" + title);
    }
    const clean = [];
    const seen = new Set();
    for (const [src, dst] of edges) {
      if (!crawled.depthOf[src] && crawled.depthOf[src] !== 0) continue;
      if (!crawled.depthOf[dst] && crawled.depthOf[dst] !== 0) continue;
      if (src === dst) continue;
      const key = src + "\0" + dst;
      if (seen.has(key)) continue;
      seen.add(key);
      outDegree.set(src, outDegree.get(src) + 1);
      inDegree.set(dst, inDegree.get(dst) + 1);
      clean.push([src, dst]);
    }
    const rankedMembers = sccs.map((component) => component.slice().sort((a, b) => {
      const diff = ppr.get(b) - ppr.get(a);
      if (diff) return diff;
      if (a < b) return -1;
      if (a > b) return 1;
      return 0;
    }));
    const nodes = titles.map((title) => {
      const sccId = membership.get(title);
      const level = sccLevels[sccId];
      const score = curriculumScore(ppr.get(title), level);
      const point = positions.get(title) || { re: 0, im: 0 };
      return {
        id: title,
        title,
        level,
        bfs_depth: crawled.depthOf[title] || 0,
        ppr: round(ppr.get(title), 8),
        score: round(score, 8),
        scc_id: sccId,
        scc_size: sccs[sccId].length,
        out_degree: outDegree.get(title),
        in_degree: inDegree.get(title),
        gateway: gateways.has(title),
        x: round(point.re, 6),
        y: round(point.im, 6),
        wikipedia: wikiUrl(title),
      };
    });
    nodes.sort((a, b) => a.level - b.level || b.score - a.score || (a.title < b.title ? -1 : a.title > b.title ? 1 : 0));
    const sccDocs = rankedMembers.map((members, id) => ({
      id,
      size: members.length,
      level: sccLevels[id],
      members,
      ppr: round(members.reduce((sum, title) => sum + ppr.get(title), 0), 8),
    }));
    const levelCounts = {};
    for (const node of nodes) {
      const key = String(node.level);
      levelCounts[key] = (levelCounts[key] || 0) + 1;
    }
    const multi = sccDocs.filter((scc) => scc.size > 1);
    let mass = 0;
    for (const title of titles) {
      if (sccLevels[membership.get(title)] <= 3) mass += ppr.get(title);
    }
    if (Math.abs(mass - 1) < 1e-9) mass = 1;
    const stats = {
      nodes: nodes.length,
      edges: clean.length,
      scc_count: sccDocs.length,
      singleton_sccs: sccDocs.length - multi.length,
      multi_sccs: multi.length,
      largest_scc: sccDocs.reduce((max, scc) => Math.max(max, scc.size), 0),
      max_level: sccLevels.length ? Math.max(...sccLevels) : 0,
      level_counts: levelCounts,
      dag_edges: dagEdges.length,
      compression: nodes.length ? round(1 - sccDocs.length / nodes.length, 6) : 0,
      gateways: gateways.size,
      ppr_mass_to_level_3: round(mass, 8),
      avg_out_degree: nodes.length ? round(clean.length / nodes.length, 6) : 0,
    };
    return {
      seed: crawled.seed,
      requested_seed: crawled.requestedSeed,
      generated_at: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
      params: Object.assign({
        walks: options.nWalks,
        walk_len: options.walkLen,
        alpha: options.alpha,
        tau: options.tau,
        sparsify_k: 0,
        source: "en.wikipedia.org mainspace outlinks in prose order",
        edge_meaning: "u → v means Wikipedia page u links to v",
        score: "PPR / (1 + topological level)",
      }, crawled.params),
      stats,
      nodes,
      edges: clean.map(([src, dst]) => ({ source: src, target: dst, tree: tree.has(src + "\0" + dst) })),
      sccs: sccDocs,
      dag_edges: dagEdges.map(([source, target]) => ({ source, target })),
    };
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function wikiGet(params, fetchImpl) {
    const url = new URL(API);
    const query = Object.assign({
      format: "json",
      formatversion: "2",
      origin: "*",
      maxlag: "5",
    }, params);
    for (const [key, value] of Object.entries(query)) url.searchParams.set(key, value);
    let last = null;
    for (let attempt = 0; attempt < 4; attempt += 1) {
      if (attempt) await sleep(Math.min(8000, 500 * 2 ** attempt));
      const response = await fetchImpl(url.toString(), { headers: { Accept: "application/json" } });
      if (response.status === 429 || response.status >= 502) {
        last = new Error("Wikipedia HTTP " + response.status);
        continue;
      }
      if (!response.ok) throw new Error("Wikipedia HTTP " + response.status);
      const payload = await response.json();
      if (payload.error) {
        if (payload.error.code === "maxlag") {
          last = new Error(payload.error.info || "maxlag");
          continue;
        }
        throw new Error(payload.error.info || payload.error.code);
      }
      return payload;
    }
    throw last || new Error("Wikipedia request failed");
  }

  async function queryPages(titles, content, fetchImpl) {
    const merged = { pages: [], redirects: [], normalized: [] };
    if (!titles.length) return merged;
    let params = {
      action: "query",
      redirects: "1",
      titles: titles.join("|"),
    };
    if (content) {
      params.prop = "revisions";
      params.rvprop = "content";
      params.rvslots = "main";
    }
    while (true) {
      const payload = await wikiGet(params, fetchImpl);
      const query = payload.query || {};
      merged.pages.push(...(query.pages || []));
      merged.redirects.push(...(query.redirects || []));
      merged.normalized.push(...(query.normalized || []));
      if (!payload.continue) return merged;
      params = Object.assign({}, params, payload.continue);
    }
  }

  function canonMap(data, batch) {
    const alias = new Map();
    for (const row of data.normalized) alias.set(row.from, row.to);
    for (const row of data.redirects) alias.set(row.from, row.to);
    const live = new Map();
    for (const page of data.pages) {
      if (page.missing || page.invalid) continue;
      live.set(page.title, page.title);
      live.set(page.title.toLowerCase(), page.title);
    }
    const out = new Map();
    for (const title of batch) {
      let current = title;
      const seen = new Set();
      while (alias.has(current) && !seen.has(current)) {
        seen.add(current);
        current = alias.get(current);
      }
      out.set(title, live.get(current) || live.get(current.toLowerCase()) || null);
    }
    return out;
  }

  async function resolveTitles(titles, fetchImpl) {
    const out = new Map();
    for (let start = 0; start < titles.length; start += 40) {
      const batch = titles.slice(start, start + 40);
      const data = await queryPages(batch, false, fetchImpl);
      const mapped = canonMap(data, batch);
      for (const [title, canon] of mapped) out.set(title, canon);
      await sleep(40);
    }
    return out;
  }

  async function crawlWikipedia(seed, opts, hooks) {
    const options = Object.assign({ depth: 2, fanout: 15, maxNodes: 150 }, opts || {});
    const fetchImpl = (hooks && hooks.fetch) || fetch;
    const onProgress = (hooks && hooks.onProgress) || function () {};
    const requested = String(seed || "").trim();
    if (!requested) throw new Error("Seed title is empty");
    if (options.depth < 0 || options.fanout < 1 || options.maxNodes < 1) {
      throw new Error("Depth, fanout, and max articles must be positive");
    }
    const resolved = await resolveTitles([requested], fetchImpl);
    const canonical = resolved.get(requested);
    if (!canonical) throw new Error("No English Wikipedia article for “" + requested + "”");
    const depthOf = { [canonical]: 0 };
    const parent = { [canonical]: null };
    const edges = [];
    const edgeSeen = new Set();
    const expand = [canonical];
    onProgress("Seed resolved to " + canonical, 1, options.maxNodes);

    while (expand.length && Object.keys(depthOf).length < options.maxNodes) {
      const batch = [];
      while (expand.length && batch.length < 6) {
        const title = expand.shift();
        if (depthOf[title] >= options.depth) continue;
        batch.push(title);
      }
      if (!batch.length) break;
      onProgress("Reading " + batch[0], Object.keys(depthOf).length, options.maxNodes);
      const data = await queryPages(batch, true, fetchImpl);
      const texts = new Map();
      for (const page of data.pages) {
        if (page.missing || page.invalid) continue;
        const revision = (page.revisions || [])[0] || {};
        const slot = revision.slots && revision.slots.main;
        const content = (slot && slot.content) || revision.content || "";
        texts.set(page.title, content);
      }
      const extracted = new Map();
      const toResolve = [];
      const cap = Math.max(options.fanout * 3, options.fanout);
      for (const title of batch) {
        const links = extractProseLinks(texts.get(title) || "", cap);
        extracted.set(title, links);
        toResolve.push(...links);
      }
      const uniq = [];
      const seenResolve = new Set();
      for (const title of toResolve) {
        const key = title.toLowerCase();
        if (seenResolve.has(key)) continue;
        seenResolve.add(key);
        uniq.push(title);
      }
      const canonOf = uniq.length ? await resolveTitles(uniq, fetchImpl) : new Map();
      for (const title of batch) {
        const kept = [];
        const keptKeys = new Set();
        for (const link of extracted.get(title)) {
          const target = canonOf.get(link);
          if (!target || target === title) continue;
          const key = target.toLowerCase();
          if (keptKeys.has(key)) continue;
          keptKeys.add(key);
          kept.push(target);
          if (kept.length >= options.fanout) break;
        }
        for (const target of kept) {
          const pair = title + "\0" + target;
          if (!edgeSeen.has(pair)) {
            edgeSeen.add(pair);
            edges.push([title, target]);
          }
          if (!(target in depthOf)) {
            if (Object.keys(depthOf).length >= options.maxNodes) break;
            depthOf[target] = depthOf[title] + 1;
            parent[target] = title;
            expand.push(target);
          }
        }
      }
      onProgress(Object.keys(depthOf).length + " articles in the graph", Object.keys(depthOf).length, options.maxNodes);
      await sleep(40);
    }
    return {
      seed: canonical,
      requestedSeed: requested,
      depthOf,
      parent,
      edges,
      params: { depth: options.depth, fanout: options.fanout, max_nodes: options.maxNodes },
    };
  }

  function diskGeodesic(x1, y1, x2, y2) {
    const det = x1 * y2 - x2 * y1;
    if (Math.abs(det) < 1e-8) return [[x1, y1], [x2, y2]];
    const rhs1 = (1 + x1 * x1 + y1 * y1) / 2;
    const rhs2 = (1 + x2 * x2 + y2 * y2) / 2;
    const cx = (rhs1 * y2 - rhs2 * y1) / det;
    const cy = (x1 * rhs2 - x2 * rhs1) / det;
    const radius2 = cx * cx + cy * cy - 1;
    if (radius2 <= 1e-10) return [[x1, y1], [x2, y2]];
    const radius = Math.sqrt(radius2);
    const a1 = Math.atan2(y1 - cy, x1 - cx);
    let delta = Math.atan2(y2 - cy, x2 - cx) - a1;
    while (delta <= -Math.PI) delta += 2 * Math.PI;
    while (delta > Math.PI) delta -= 2 * Math.PI;
    const steps = 10;
    const points = [];
    for (let i = 0; i <= steps; i += 1) {
      const angle = a1 + delta * (i / steps);
      points.push([cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)]);
    }
    return points;
  }

  function diskToHalfPlane(x, y) {
    const czr = x;
    const czi = -y;
    const nr = 1 + czr;
    const ni = czi;
    const dr = 1 - czr;
    const di = -czi;
    const denom = dr * dr + di * di;
    if (denom < 1e-12) return null;
    const qr = (nr * dr + ni * di) / denom;
    const qi = (ni * dr - nr * di) / denom;
    return { x: -qi, y: -qr };
  }

  return {
    extractProseLinks,
    iterativeTarjan,
    preparedAdjacency,
    analyze,
    curriculumScore,
    assembleCurriculum,
    crawlWikipedia,
    sarkarEmbed,
    diskGeodesic,
    diskToHalfPlane,
    wikiUrl,
  };
});
