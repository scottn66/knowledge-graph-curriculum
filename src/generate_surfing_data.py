#!/usr/bin/env python3
"""
Generate a poincare_art_data.json for SURFING as the seed topic.

Since we don't have ref_counts.db for surfing, we build a plausible
knowledge graph from domain knowledge:
  - ~150 concept nodes across surfing-related domains
  - PPR-like scores (seed = "Surfing", decay with graph distance)
  - Domain classification
  - Edges reflecting real reference relationships
  - Poincaré disk coordinates: r = f(PPR), θ = f(domain sector)
"""

import json, math, random
random.seed(2026)

# ═══════════════════════════════════════════════════════════════
# 1. Define the knowledge graph
# ═══════════════════════════════════════════════════════════════

# Domains and their angular sectors (degrees)
DOMAINS = {
    'core_surfing':       0,
    'surf_culture':       36,
    'oceanography':       72,
    'board_design':       108,
    'competition':        144,
    'geography':          180,
    'weather_climate':    216,
    'physiology':         252,
    'history':            288,
    'environment':        324,
}

# Node definitions: (name, domain, tier)
# tier 0 = seed, tier 1 = immediate neighbors, tier 2 = 2-hop, tier 3 = periphery
CONCEPT_NODES = [
    # ─── Core Surfing (seed cluster) ───
    ("Surfing", "core_surfing", 0),
    ("Surf break", "core_surfing", 1),
    ("Wave riding", "core_surfing", 1),
    ("Paddle out", "core_surfing", 1),
    ("Pop-up (surfing)", "core_surfing", 1),
    ("Duck dive", "core_surfing", 1),
    ("Tube riding", "core_surfing", 1),
    ("Cutback", "core_surfing", 1),
    ("Bottom turn", "core_surfing", 1),
    ("Aerial (surfing)", "core_surfing", 2),
    ("Floater (surfing)", "core_surfing", 2),
    ("Carving (surfing)", "core_surfing", 2),
    ("Wipeout", "core_surfing", 1),
    ("Lineup (surfing)", "core_surfing", 2),
    ("Surf leash", "core_surfing", 2),
    ("Wetsuit", "core_surfing", 1),

    # ─── Surf Culture ───
    ("Surf culture", "surf_culture", 1),
    ("Surf music", "surf_culture", 2),
    ("The Beach Boys", "surf_culture", 2),
    ("Dick Dale", "surf_culture", 3),
    ("Surf film", "surf_culture", 2),
    ("The Endless Summer", "surf_culture", 2),
    ("Point Break (1991 film)", "surf_culture", 3),
    ("Surf art", "surf_culture", 3),
    ("Surf photography", "surf_culture", 2),
    ("Surfing subculture", "surf_culture", 2),
    ("Localism (surfing)", "surf_culture", 2),
    ("Soul surfer", "surf_culture", 3),
    ("Surf slang", "surf_culture", 2),
    ("Aloha spirit", "surf_culture", 3),
    ("Tiki culture", "surf_culture", 3),

    # ─── Oceanography ───
    ("Ocean surface wave", "oceanography", 1),
    ("Swell (ocean)", "oceanography", 1),
    ("Wave height", "oceanography", 2),
    ("Wave period", "oceanography", 2),
    ("Significant wave height", "oceanography", 2),
    ("Breaking wave", "oceanography", 1),
    ("Rip current", "oceanography", 1),
    ("Tide", "oceanography", 2),
    ("Tidal range", "oceanography", 3),
    ("Bathymetry", "oceanography", 2),
    ("Continental shelf", "oceanography", 3),
    ("Kelp forest", "oceanography", 3),
    ("Coral reef", "oceanography", 2),
    ("Ocean current", "oceanography", 2),
    ("Undertow (water waves)", "oceanography", 2),
    ("Fetch (geography)", "oceanography", 3),
    ("Beaufort scale", "oceanography", 3),

    # ─── Board Design & Equipment ───
    ("Surfboard", "board_design", 1),
    ("Longboard (surfing)", "board_design", 1),
    ("Shortboard", "board_design", 1),
    ("Fish (surfboard)", "board_design", 2),
    ("Gun (surfboard)", "board_design", 2),
    ("Foam board (surfing)", "board_design", 2),
    ("Surfboard shaping", "board_design", 2),
    ("Surfboard fin", "board_design", 2),
    ("Thruster (surfboard)", "board_design", 2),
    ("Twin fin", "board_design", 3),
    ("Polyurethane foam", "board_design", 3),
    ("Epoxy", "board_design", 3),
    ("Fiberglass", "board_design", 3),
    ("Wax (surfing)", "board_design", 2),
    ("Traction pad", "board_design", 3),
    ("Hydrofoil", "board_design", 2),

    # ─── Competition ───
    ("World Surf League", "competition", 1),
    ("Championship Tour (surfing)", "competition", 2),
    ("Big wave surfing", "competition", 1),
    ("Eddie Aikau", "competition", 2),
    ("Kelly Slater", "competition", 1),
    ("Laird Hamilton", "competition", 2),
    ("Stephanie Gilmore", "competition", 2),
    ("John John Florence", "competition", 2),
    ("Bethany Hamilton", "competition", 2),
    ("Duke Kahanamoku", "competition", 1),
    ("Tow-in surfing", "competition", 2),
    ("Surfing at the Olympics", "competition", 2),
    ("2020 Summer Olympics", "competition", 3),
    ("Surf forecasting", "competition", 2),
    ("Judging (surfing)", "competition", 3),

    # ─── Geography (surf spots) ───
    ("Pipeline (surf spot)", "geography", 1),
    ("Teahupo'o", "geography", 2),
    ("Mavericks (California)", "geography", 2),
    ("Nazaré (waves)", "geography", 2),
    ("Jeffreys Bay", "geography", 2),
    ("Bells Beach", "geography", 2),
    ("Uluwatu (surf spot)", "geography", 3),
    ("Gold Coast (Australia)", "geography", 2),
    ("North Shore (Oahu)", "geography", 1),
    ("Santa Cruz, California", "geography", 3),
    ("Byron Bay", "geography", 3),
    ("Hossegor", "geography", 3),
    ("Mentawai Islands", "geography", 3),
    ("Cloudbreak", "geography", 3),
    ("Trestles (surfing)", "geography", 3),

    # ─── Weather & Climate ───
    ("Surf forecasting", "weather_climate", 2),
    ("Wind wave", "weather_climate", 2),
    ("Trade winds", "weather_climate", 2),
    ("Tropical cyclone", "weather_climate", 3),
    ("El Niño–Southern Oscillation", "weather_climate", 2),
    ("Storm surge", "weather_climate", 3),
    ("Offshore wind", "weather_climate", 2),
    ("Sea surface temperature", "weather_climate", 3),
    ("Marine weather forecasting", "weather_climate", 3),
    ("Buoy", "weather_climate", 3),
    ("NOAA", "weather_climate", 3),

    # ─── Physiology & Fitness ───
    ("Swimming", "physiology", 1),
    ("Drowning", "physiology", 2),
    ("Hypothermia", "physiology", 2),
    ("Surfer's ear", "physiology", 2),
    ("Balance (ability)", "physiology", 2),
    ("Proprioception", "physiology", 3),
    ("Core stability", "physiology", 3),
    ("Shoulder (anatomy)", "physiology", 3),
    ("Water safety", "physiology", 2),
    ("Lifeguard", "physiology", 2),
    ("Surf lifesaving", "physiology", 2),
    ("CPR", "physiology", 3),
    ("Skin cancer", "physiology", 3),
    ("Sunscreen", "physiology", 3),

    # ─── History ───
    ("History of surfing", "history", 1),
    ("Ancient Hawaii", "history", 1),
    ("Polynesians", "history", 2),
    ("James Cook", "history", 3),
    ("Tom Blake (surfer)", "history", 2),
    ("Miki Dora", "history", 3),
    ("Greg Noll", "history", 3),
    ("California surf boom", "history", 2),
    ("Surf industry", "history", 2),
    ("Quiksilver", "history", 3),
    ("Billabong (company)", "history", 3),
    ("Rip Curl", "history", 3),
    ("Hawaiian sovereignty movement", "history", 3),
    ("Waikiki", "history", 2),

    # ─── Environment ───
    ("Marine conservation", "environment", 2),
    ("Ocean pollution", "environment", 2),
    ("Plastic pollution", "environment", 2),
    ("Surfrider Foundation", "environment", 2),
    ("Beach erosion", "environment", 2),
    ("Coastal management", "environment", 3),
    ("Artificial reef", "environment", 2),
    ("Shark", "environment", 2),
    ("Shark attack", "environment", 2),
    ("Shark net", "environment", 3),
    ("Jellyfish", "environment", 3),
    ("Stingray", "environment", 3),
    ("Ecosystem", "environment", 3),
    ("Sustainable surfing", "environment", 3),
]

# De-duplicate by name
seen = set()
unique_nodes = []
for name, domain, tier in CONCEPT_NODES:
    if name not in seen:
        seen.add(name)
        unique_nodes.append((name, domain, tier))
CONCEPT_NODES = unique_nodes

# Assign IDs
nodes = []
name_to_id = {}
for i, (name, domain, tier) in enumerate(CONCEPT_NODES):
    nid = i + 1
    name_to_id[name] = nid
    nodes.append({"id": nid, "name": name, "domain": domain, "tier": tier})

# ═══════════════════════════════════════════════════════════════
# 2. Generate edges (plausible reference structure)
# ═══════════════════════════════════════════════════════════════

EDGE_DEFS = [
    # Seed to tier-1
    ("Surfing", "Surf break"), ("Surfing", "Wave riding"), ("Surfing", "Surfboard"),
    ("Surfing", "Ocean surface wave"), ("Surfing", "Surf culture"), ("Surfing", "History of surfing"),
    ("Surfing", "World Surf League"), ("Surfing", "Pipeline (surf spot)"), ("Surfing", "Swimming"),
    ("Surfing", "Wetsuit"), ("Surfing", "Big wave surfing"), ("Surfing", "Duke Kahanamoku"),
    ("Surfing", "Kelly Slater"), ("Surfing", "North Shore (Oahu)"), ("Surfing", "Wipeout"),
    ("Surfing", "Breaking wave"), ("Surfing", "Rip current"), ("Surfing", "Swell (ocean)"),
    ("Surfing", "Longboard (surfing)"), ("Surfing", "Shortboard"),

    # Core surfing interconnections
    ("Surf break", "Breaking wave"), ("Surf break", "Bathymetry"), ("Surf break", "Coral reef"),
    ("Wave riding", "Bottom turn"), ("Wave riding", "Tube riding"), ("Wave riding", "Cutback"),
    ("Paddle out", "Duck dive"), ("Paddle out", "Lineup (surfing)"),
    ("Pop-up (surfing)", "Balance (ability)"), ("Pop-up (surfing)", "Bottom turn"),
    ("Tube riding", "Pipeline (surf spot)"), ("Tube riding", "Teahupo'o"),
    ("Cutback", "Carving (surfing)"), ("Cutback", "Bottom turn"),
    ("Aerial (surfing)", "Floater (surfing)"), ("Aerial (surfing)", "John John Florence"),
    ("Wipeout", "Rip current"), ("Wipeout", "Drowning"), ("Wipeout", "Surf leash"),
    ("Duck dive", "Wipeout"),

    # Oceanography
    ("Ocean surface wave", "Swell (ocean)"), ("Ocean surface wave", "Wave height"),
    ("Ocean surface wave", "Wave period"), ("Ocean surface wave", "Wind wave"),
    ("Swell (ocean)", "Fetch (geography)"), ("Swell (ocean)", "Significant wave height"),
    ("Swell (ocean)", "Surf forecasting"), ("Breaking wave", "Wave height"),
    ("Breaking wave", "Bathymetry"), ("Rip current", "Undertow (water waves)"),
    ("Rip current", "Water safety"), ("Tide", "Tidal range"), ("Tide", "Surf break"),
    ("Bathymetry", "Continental shelf"), ("Coral reef", "Kelp forest"),
    ("Ocean current", "El Niño–Southern Oscillation"), ("Ocean current", "Trade winds"),
    ("Wave height", "Beaufort scale"),

    # Board design
    ("Surfboard", "Longboard (surfing)"), ("Surfboard", "Shortboard"), ("Surfboard", "Fish (surfboard)"),
    ("Surfboard", "Gun (surfboard)"), ("Surfboard", "Surfboard shaping"), ("Surfboard", "Surfboard fin"),
    ("Surfboard", "Wax (surfing)"), ("Surfboard", "Foam board (surfing)"),
    ("Longboard (surfing)", "Surfboard fin"), ("Shortboard", "Thruster (surfboard)"),
    ("Shortboard", "Aerial (surfing)"), ("Fish (surfboard)", "Twin fin"),
    ("Gun (surfboard)", "Big wave surfing"), ("Surfboard shaping", "Polyurethane foam"),
    ("Surfboard shaping", "Epoxy"), ("Surfboard shaping", "Fiberglass"),
    ("Surfboard fin", "Thruster (surfboard)"), ("Surfboard fin", "Twin fin"),
    ("Hydrofoil", "Surfboard"), ("Traction pad", "Wax (surfing)"),

    # Competition
    ("World Surf League", "Championship Tour (surfing)"), ("World Surf League", "Kelly Slater"),
    ("World Surf League", "Stephanie Gilmore"), ("World Surf League", "John John Florence"),
    ("World Surf League", "Judging (surfing)"),
    ("Big wave surfing", "Tow-in surfing"), ("Big wave surfing", "Laird Hamilton"),
    ("Big wave surfing", "Eddie Aikau"), ("Big wave surfing", "Nazaré (waves)"),
    ("Big wave surfing", "Mavericks (California)"), ("Big wave surfing", "Teahupo'o"),
    ("Kelly Slater", "Pipeline (surf spot)"), ("Kelly Slater", "North Shore (Oahu)"),
    ("Duke Kahanamoku", "Ancient Hawaii"), ("Duke Kahanamoku", "Waikiki"),
    ("Duke Kahanamoku", "Swimming"), ("Surfing at the Olympics", "2020 Summer Olympics"),
    ("Surfing at the Olympics", "World Surf League"),
    ("Bethany Hamilton", "Shark attack"), ("Eddie Aikau", "North Shore (Oahu)"),
    ("Laird Hamilton", "Tow-in surfing"), ("Surf forecasting", "Buoy"),

    # Geography
    ("Pipeline (surf spot)", "North Shore (Oahu)"), ("Pipeline (surf spot)", "Breaking wave"),
    ("Teahupo'o", "Coral reef"), ("Mavericks (California)", "Santa Cruz, California"),
    ("Nazaré (waves)", "Big wave surfing"), ("Jeffreys Bay", "Surfing"),
    ("Bells Beach", "World Surf League"), ("Gold Coast (Australia)", "World Surf League"),
    ("North Shore (Oahu)", "Ancient Hawaii"), ("North Shore (Oahu)", "Eddie Aikau"),
    ("Uluwatu (surf spot)", "Mentawai Islands"), ("Byron Bay", "Gold Coast (Australia)"),
    ("Cloudbreak", "Mentawai Islands"), ("Trestles (surfing)", "World Surf League"),

    # Weather
    ("Surf forecasting", "Marine weather forecasting"), ("Surf forecasting", "NOAA"),
    ("Wind wave", "Trade winds"), ("Wind wave", "Offshore wind"),
    ("Tropical cyclone", "Storm surge"), ("Tropical cyclone", "Swell (ocean)"),
    ("El Niño–Southern Oscillation", "Sea surface temperature"),
    ("Offshore wind", "Surf break"),

    # Physiology
    ("Swimming", "Drowning"), ("Swimming", "Water safety"), ("Swimming", "Lifeguard"),
    ("Drowning", "CPR"), ("Hypothermia", "Wetsuit"), ("Hypothermia", "Sea surface temperature"),
    ("Surfer's ear", "Surfing"), ("Balance (ability)", "Proprioception"),
    ("Balance (ability)", "Core stability"), ("Water safety", "Lifeguard"),
    ("Lifeguard", "Surf lifesaving"), ("Surf lifesaving", "CPR"),
    ("Skin cancer", "Sunscreen"), ("Shoulder (anatomy)", "Paddle out"),

    # History
    ("History of surfing", "Ancient Hawaii"), ("History of surfing", "Duke Kahanamoku"),
    ("History of surfing", "Tom Blake (surfer)"), ("History of surfing", "California surf boom"),
    ("Ancient Hawaii", "Polynesians"), ("Ancient Hawaii", "Hawaiian sovereignty movement"),
    ("Polynesians", "James Cook"), ("Tom Blake (surfer)", "Surfboard shaping"),
    ("California surf boom", "Surf culture"), ("California surf boom", "Miki Dora"),
    ("California surf boom", "Greg Noll"), ("Surf industry", "Quiksilver"),
    ("Surf industry", "Billabong (company)"), ("Surf industry", "Rip Curl"),
    ("Waikiki", "Ancient Hawaii"), ("Waikiki", "Duke Kahanamoku"),

    # Culture
    ("Surf culture", "Surf music"), ("Surf culture", "Surf film"), ("Surf culture", "Surf art"),
    ("Surf culture", "Surfing subculture"), ("Surf culture", "Surf slang"),
    ("Surf music", "The Beach Boys"), ("Surf music", "Dick Dale"),
    ("Surf film", "The Endless Summer"), ("Surf film", "Point Break (1991 film)"),
    ("Surf photography", "Surf film"), ("Surf photography", "Pipeline (surf spot)"),
    ("Surfing subculture", "Localism (surfing)"), ("Surfing subculture", "Soul surfer"),
    ("Aloha spirit", "Ancient Hawaii"), ("Aloha spirit", "Surf culture"),
    ("Tiki culture", "Surf culture"),

    # Environment
    ("Marine conservation", "Ocean pollution"), ("Marine conservation", "Surfrider Foundation"),
    ("Marine conservation", "Artificial reef"), ("Ocean pollution", "Plastic pollution"),
    ("Surfrider Foundation", "Beach erosion"), ("Beach erosion", "Coastal management"),
    ("Artificial reef", "Surf break"), ("Artificial reef", "Coral reef"),
    ("Shark", "Shark attack"), ("Shark attack", "Shark net"),
    ("Shark", "Ecosystem"), ("Jellyfish", "Stingray"),
    ("Sustainable surfing", "Marine conservation"), ("Sustainable surfing", "Surf industry"),
    ("Ecosystem", "Kelp forest"), ("Ecosystem", "Coral reef"),

    # Cross-domain connections
    ("Wetsuit", "Hypothermia"), ("Wetsuit", "Surf industry"),
    ("Surf leash", "Surfboard"), ("Lineup (surfing)", "Localism (surfing)"),
    ("Tow-in surfing", "Jet ski"), # will skip if not in graph
    ("Surfing", "Marine conservation"), ("Surfing", "Surfing at the Olympics"),
    ("Surfing", "Surf forecasting"),
]

# Build edge list (skip edges with unknown nodes)
edges = []
edge_set = set()
for src, tgt in EDGE_DEFS:
    if src in name_to_id and tgt in name_to_id:
        sid, tid = name_to_id[src], name_to_id[tgt]
        if (sid, tid) not in edge_set:
            edges.append({"source": sid, "target": tid})
            edge_set.add((sid, tid))
            # Add reverse for some (bidirectional references)
            if random.random() < 0.3 and (tid, sid) not in edge_set:
                edges.append({"source": tid, "target": sid})
                edge_set.add((tid, sid))

# ═══════════════════════════════════════════════════════════════
# 3. Compute pseudo-PPR via BFS decay from seed
# ═══════════════════════════════════════════════════════════════

# Build adjacency
from collections import defaultdict, deque

adj = defaultdict(list)
for e in edges:
    adj[e['source']].append(e['target'])
    adj[e['target']].append(e['source'])  # undirected for PPR proxy

seed_id = name_to_id["Surfing"]
alpha = 0.15

# Random walk PPR approximation
ppr = defaultdict(float)
n_walks = 50000
walk_len = 20

for _ in range(n_walks):
    cur = seed_id
    for step in range(walk_len):
        ppr[cur] += 1.0
        if random.random() < alpha or not adj[cur]:
            cur = seed_id  # restart
        else:
            cur = random.choice(adj[cur])

# Normalize
total = sum(ppr.values())
for k in ppr:
    ppr[k] /= total

# Compute degree
degree = defaultdict(int)
for e in edges:
    degree[e['source']] += 1
    degree[e['target']] += 1

# ═══════════════════════════════════════════════════════════════
# 4. Assign Poincaré disk coordinates
# ═══════════════════════════════════════════════════════════════

ppr_max = max(ppr[n['id']] for n in nodes) if nodes else 1.0
ppr_min = min(ppr[n['id']] for n in nodes if ppr[n['id']] > 0) if nodes else 1e-6

for n in nodes:
    nid = n['id']
    p = ppr.get(nid, ppr_min * 0.5)

    # Radial: high PPR → near center, low PPR → near boundary
    if p > 0 and ppr_max > 0:
        log_ratio = math.log(p / ppr_max)
        # Map log_ratio ∈ [log(ppr_min/ppr_max), 0] → r ∈ [0.92, 0.05]
        log_range = abs(math.log(ppr_min / ppr_max)) if ppr_min > 0 else 10
        r = 0.05 + 0.87 * min(1.0, abs(log_ratio) / log_range)
    else:
        r = 0.92

    # Angular: domain sector + within-domain jitter
    sector_center = DOMAINS.get(n['domain'], 0)
    sector_width = 32  # degrees, with small gaps between sectors
    # Rank within domain for angular spread
    same_domain = [nd for nd in nodes if nd['domain'] == n['domain']]
    same_domain.sort(key=lambda nd: ppr.get(nd['id'], 0), reverse=True)
    rank = next(i for i, nd in enumerate(same_domain) if nd['id'] == nid)
    n_in_domain = len(same_domain)
    angular_offset = (rank / max(1, n_in_domain - 1) - 0.5) * sector_width if n_in_domain > 1 else 0
    theta_deg = sector_center + angular_offset + random.gauss(0, 2)
    theta = math.radians(theta_deg)

    n['x'] = round(r * math.cos(theta), 5)
    n['y'] = round(r * math.sin(theta), 5)
    n['r'] = round(r, 4)
    n['ppr'] = round(p, 8)
    n['degree'] = degree.get(nid, 0)

    # Remove tier (internal)
    if 'tier' in n:
        del n['tier']

# ═══════════════════════════════════════════════════════════════
# 5. Write output
# ═══════════════════════════════════════════════════════════════

output = {"nodes": nodes, "edges": edges}
outpath = "/sessions/charming-exciting-cori/poincare_art_data_surfing.json"
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Generated {len(nodes)} nodes, {len(edges)} edges")
print(f"Written to {outpath}")
print(f"PPR range: {min(n['ppr'] for n in nodes):.6f} – {max(n['ppr'] for n in nodes):.6f}")
print(f"Degree range: {min(n['degree'] for n in nodes)} – {max(n['degree'] for n in nodes)}")
print(f"Domains: {sorted(set(n['domain'] for n in nodes))}")
