"use strict";

const assert = require("assert");
const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");
const KGC = require("../docs/curriculum.js");

const SAMPLE = `
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
`;

assert.deepStrictEqual(KGC.extractProseLinks(SAMPLE), [
  "Light", "Refraction", "Lens", "Snell's law", "Photon",
]);
assert.deepStrictEqual(KGC.extractProseLinks(SAMPLE, 3), ["Light", "Refraction", "Lens"]);
assert.deepStrictEqual(
  KGC.extractProseLinks("See [[The Lord of the Rings: The Fellowship of the Ring]] and [[fr:Optique]]."),
  ["The Lord of the Rings: The Fellowship of the Ring"],
);
assert.strictEqual(KGC.curriculumScore(0.2, 3), 0.05);

const doc = JSON.parse(fs.readFileSync(path.join(__dirname, "../docs/data/optics_graph.json"), "utf8"));
const analyzed = KGC.analyze(
  doc.nodes.map((node) => node.title),
  doc.edges.map((edge) => [edge.source, edge.target]),
);
for (const node of doc.nodes) {
  const got = analyzed.byTitle[node.title];
  assert.strictEqual(got.sccId, node.scc_id, node.title + " scc");
  assert.strictEqual(got.level, node.level, node.title + " level");
  assert.strictEqual(got.sccSize, node.scc_size, node.title + " size");
}
assert.strictEqual(analyzed.dagEdges.length, doc.dag_edges.length);

const py = JSON.parse(execFileSync("python3", ["-c", `
import json, sys
sys.path.insert(0, "src")
from kgc.embed import sarkar_embed, disk_to_half_plane
children = {"root": ["a", "b"], "a": ["a1"], "b": [], "a1": []}
pos = sarkar_embed("root", children, tau=1.0)
half = disk_to_half_plane(0.2+0.3j)
print(json.dumps({
  "pos": {k: [v.real, v.imag] for k, v in pos.items()},
  "half": [half.real, half.imag],
}))
`], { cwd: path.join(__dirname, "..") }).toString());

const children = new Map([
  ["root", ["a", "b"]],
  ["a", ["a1"]],
  ["b", []],
  ["a1", []],
]);
const pos = KGC.sarkarEmbed("root", children, 1);
for (const [name, xy] of Object.entries(py.pos)) {
  const got = pos.get(name);
  assert.ok(Math.abs(got.re - xy[0]) < 1e-9, name + " re");
  assert.ok(Math.abs(got.im - xy[1]) < 1e-9, name + " im");
}
const half = KGC.diskToHalfPlane(0.2, 0.3);
assert.ok(Math.abs(half.x - py.half[0]) < 1e-9);
assert.ok(Math.abs(half.y - py.half[1]) < 1e-9);
assert.ok(Math.abs(pos.get("root").re) < 1e-12);

const crawled = {
  seed: "Tide",
  requestedSeed: "Tides",
  depthOf: { Tide: 0, Wave: 1, Moon: 1 },
  parent: { Tide: null, Wave: "Tide", Moon: "Tide" },
  edges: [["Tide", "Wave"], ["Tide", "Moon"], ["Wave", "Tide"]],
  params: { depth: 1, fanout: 2, max_nodes: 10 },
};
const built = KGC.assembleCurriculum(crawled, {
  nWalks: 50,
  walkLen: 5,
  alpha: 0.15,
  random: () => 0.5,
});
assert.strictEqual(built.stats.nodes, 3);
assert.ok(Math.abs(built.nodes.reduce((sum, node) => sum + node.ppr, 0) - 1) < 1e-6);
const tide = built.nodes.find((node) => node.title === "Tide");
const wave = built.nodes.find((node) => node.title === "Wave");
assert.strictEqual(tide.scc_id, wave.scc_id);
assert.strictEqual(tide.level, 0);
assert.ok(Math.hypot(tide.x, tide.y) < 1e-9);

console.log("curriculum.js tests passed");
