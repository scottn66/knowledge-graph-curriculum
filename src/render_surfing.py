#!/usr/bin/env python3
"""
Render surfing knowledge graph as Poincaré disk + half-plane SVGs.
Uses hyperbolic_engine.py with surfing-specific domain colors and sectors.
"""

import json, math, random
from hyperbolic_engine import (
    Point, Ideal, Line, Hypercycle, Circle, Horocycle, Transform,
    SVGRenderer, _clamp_to_disk
)

random.seed(42)

# ─── Surfing domain palette (ocean-inspired) ─────────────────────────
DOMAIN_COLORS = {
    'core_surfing':    '#00d4ff',   # cyan (wave crest)
    'surf_culture':    '#ff6b6b',   # coral
    'oceanography':    '#1e90ff',   # dodger blue
    'board_design':    '#ffd700',   # gold (board resin)
    'competition':     '#ff8c42',   # sunset orange
    'geography':       '#96f7a0',   # mint (islands)
    'weather_climate': '#c0c0ff',   # lavender-grey (storm)
    'physiology':      '#ff9ff3',   # pink
    'history':         '#d4a574',   # sand
    'environment':     '#7fffd4',   # aquamarine
}

DOMAIN_SECTORS = {
    'core_surfing':    0,
    'surf_culture':    36,
    'oceanography':    72,
    'board_design':    108,
    'competition':     144,
    'geography':       180,
    'weather_climate': 216,
    'physiology':      252,
    'history':         288,
    'environment':     324,
}

def domain_color(d):
    return DOMAIN_COLORS.get(d, '#888888')

def lighten(hex_color, factor=0.3):
    h = hex_color.lstrip('#')
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    r, g, b = int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255
    r = r + (1-r)*factor
    g = g + (1-g)*factor
    b = b + (1-b)*factor
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


# ─── Load data ──────────────────────────────────────────────────────
with open('/sessions/charming-exciting-cori/poincare_art_data_surfing.json') as f:
    data = json.load(f)

nodes = data['nodes']
edges = data['edges']
node_map = {n['id']: n for n in nodes}

h_points = {}
for n in nodes:
    x, y = _clamp_to_disk(n['x'], n['y'], margin=0.02)
    h_points[n['id']] = Point(x, y)

ppr_values = [n['ppr'] for n in nodes]
max_ppr = max(ppr_values) if ppr_values else 1.0
min_ppr = min(p for p in ppr_values if p > 0) if any(p > 0 for p in ppr_values) else 1e-8
max_degree = max(n['degree'] for n in nodes) if nodes else 1

edge_set = set()
valid_edges = []
for e in edges:
    s, t = e['source'], e['target']
    if s in node_map and t in node_map and s != t:
        key = (min(s, t), max(s, t))
        if key not in edge_set:
            edge_set.add(key)
            valid_edges.append(e)

def edge_importance(e):
    s_ppr = node_map[e['source']]['ppr']
    t_ppr = node_map[e['target']]['ppr']
    return math.sqrt(s_ppr * t_ppr)

valid_edges.sort(key=edge_importance, reverse=True)
top_edges = valid_edges[:30]
mid_edges = valid_edges[30:90]
dim_edges = valid_edges[90:]

domain_sentinels = {}
for n in nodes:
    d = n['domain']
    r = math.hypot(n['x'], n['y'])
    if d not in domain_sentinels or r > domain_sentinels[d][1]:
        domain_sentinels[d] = (n['id'], r)

print(f"Nodes: {len(nodes)}, Edges: {len(valid_edges)}")

# ═══════════════════════════════════════════════════════════════════════
# Poincaré Disk
# ═══════════════════════════════════════════════════════════════════════

def render_disk():
    svg = SVGRenderer(2.2, 2.2, origin='center')

    svg.add_def('''
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="0.012" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="softglow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="0.025" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <radialGradient id="diskbg" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#02080f"/>
      <stop offset="60%" stop-color="#061828"/>
      <stop offset="100%" stop-color="#0b2845"/>
    </radialGradient>
    <radialGradient id="boundary_glow" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse">
      <stop offset="80%" stop-color="transparent"/>
      <stop offset="95%" stop-color="rgba(64,180,255,0.18)"/>
      <stop offset="100%" stop-color="rgba(120,220,255,0.32)"/>
    </radialGradient>
    ''')

    for domain in DOMAIN_SECTORS:
        color = DOMAIN_COLORS.get(domain, '#888888')
        svg.add_def(f'''
    <radialGradient id="sector_{domain}" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="transparent"/>
      <stop offset="60%" stop-color="{color}" stop-opacity="0.02"/>
      <stop offset="100%" stop-color="{color}" stop-opacity="0.08"/>
    </radialGradient>''')

    svg.circle(0, 0, 1.05, fill='#02060c')
    svg.circle(0, 0, 1.0, fill='url(#diskbg)')
    svg.circle(0, 0, 1.0, fill='url(#boundary_glow)')

    # Domain wedges
    for domain, base_angle in DOMAIN_SECTORS.items():
        color = DOMAIN_COLORS.get(domain, '#888')
        a1 = math.radians(base_angle - 18)
        a2 = math.radians(base_angle + 18)
        x1, y1 = math.cos(a1), math.sin(a1)
        x2, y2 = math.cos(a2), math.sin(a2)
        d = f"M 0,0 L {x1},{y1} A 1,1 0 0,1 {x2},{y2} Z"
        svg.path(d, fill=color, opacity=0.06)

    for r_val in [0.2, 0.4, 0.6, 0.8]:
        svg.circle(0, 0, r_val, fill='none', stroke='#ffffff', stroke_width=0.002, opacity=0.10)

    # Horocycles
    for domain, (nid, radius) in domain_sentinels.items():
        p = h_points[nid]
        color = domain_color(domain)
        try:
            horo = Horocycle.from_closest_point(p, surround_origin=False)
            svg.draw_horocycle(horo, hwidth=0.008, fill=color, opacity=0.32)
        except Exception:
            pass

    # Dim edges
    svg.group_start(opacity='0.08')
    for e in dim_edges:
        s, t = e['source'], e['target']
        p1, p2 = h_points[s], h_points[t]
        try:
            line = Line.from_points(p1.x, p1.y, p2.x, p2.y, segment=True)
            c = domain_color(node_map[s]['domain'])
            svg.draw_geodesic(line, hwidth=0.003, fill=c, stroke=c, stroke_width=0.003)
        except Exception:
            pass
    svg.group_end()

    # Mid edges
    svg.group_start(opacity='0.25')
    for e in mid_edges:
        s, t = e['source'], e['target']
        p1, p2 = h_points[s], h_points[t]
        try:
            line = Line.from_points(p1.x, p1.y, p2.x, p2.y, segment=True)
            c = domain_color(node_map[s]['domain'])
            svg.draw_geodesic(line, hwidth=0.005, fill=c, stroke=c, stroke_width=0.005)
        except Exception:
            pass
    svg.group_end()

    # Top edges (hypercycle ribbons)
    svg.group_start(filter='url(#softglow)')
    for e in top_edges:
        s, t = e['source'], e['target']
        s_data, t_data = node_map[s], node_map[t]
        p1, p2 = h_points[s], h_points[t]
        try:
            line = Line.from_points(p1.x, p1.y, p2.x, p2.y, segment=True)
            geo_ppr = math.sqrt(s_data['ppr'] * t_data['ppr'])
            offset = 0.08 + 0.4 * (geo_ppr / max_ppr)
            c = domain_color(s_data['domain'])
            c_light = lighten(c, 0.3)
            hc1 = Hypercycle.from_hypercycle_offset(line, offset)
            svg.draw_hypercycle(hc1, hwidth=0.025, fill=c, opacity=0.3)
            hc2 = Hypercycle.from_hypercycle_offset(line, offset * 0.5)
            svg.draw_hypercycle(hc2, hwidth=0.008, fill=c_light, opacity=0.7)
            hc3 = Hypercycle.from_hypercycle_offset(line, -offset)
            svg.draw_hypercycle(hc3, hwidth=0.018, fill=c, opacity=0.2)
            svg.draw_geodesic(line, hwidth=0.004, fill='white', stroke='white', stroke_width=0.004, opacity=0.5)
        except Exception:
            pass
    svg.group_end()

    # Degree halos
    top_nodes = sorted(nodes, key=lambda n: n['degree'], reverse=True)[:25]
    for n in top_nodes:
        p = h_points[n['id']]
        h_r = 0.05 + 0.15 * math.log(1 + n['degree']) / max(math.log(1 + max_degree), 1)
        c = domain_color(n['domain'])
        try:
            circ = Circle.from_center_radius(p, h_r)
            svg.draw_circle(circ, fill=c, opacity=0.12)
            circ2 = Circle.from_center_radius(p, h_r * 0.6)
            svg.draw_circle(circ2, fill='none', stroke=c, stroke_width=0.004, opacity=0.3)
        except Exception:
            pass

    # Node points (with data attributes for interactivity)
    for n in sorted(nodes, key=lambda x: x['ppr']):
        p = h_points[n['id']]
        ppr_norm = (n['ppr'] - min_ppr) / (max_ppr - min_ppr) if max_ppr > min_ppr else 0.5
        ppr_norm = max(0, min(1, ppr_norm))
        radius = 0.008 + 0.025 * ppr_norm
        c = domain_color(n['domain'])
        name_safe = n['name'].replace('"', '&quot;').replace('&', '&amp;')
        svg.raw(
            f'<circle cx="{p.x}" cy="{p.y}" r="{radius}" fill="{c}" opacity="0.9" '
            f'class="kg-node" data-id="{n["id"]}" data-name="{name_safe}" '
            f'data-domain="{n["domain"]}" data-ppr="{n["ppr"]:.6f}" '
            f'data-degree="{n["degree"]}" />'
        )
        svg.draw_point(p, radius=radius*0.4, fill='white', opacity=0.6)

    # Labels for top PPR
    top20 = sorted(nodes, key=lambda n: n['ppr'], reverse=True)[:20]
    for n in top20:
        p = h_points[n['id']]
        name = n['name'].replace('_', ' ')
        if len(name) > 20:
            name = name[:18] + '..'
        ppr_norm = (n['ppr'] - min_ppr) / (max_ppr - min_ppr) if max_ppr > min_ppr else 0.5
        fs = 0.022 + 0.014 * ppr_norm
        c = lighten(domain_color(n['domain']), 0.4)
        svg.text(p.x, p.y - 0.04, name, font_size=fs, fill=c, opacity=0.88)

    svg.circle(0, 0, 1.0, fill='none', stroke='#4488cc', stroke_width=0.006, opacity=0.55)

    svg.text(0, -1.07, 'POINCARÉ DISK · Surfing Knowledge Graph',
             font_size=0.035, fill='#88ccff', opacity=0.75)

    # Legend
    lx, ly = -1.05, 0.62
    for i, (domain, color) in enumerate(DOMAIN_COLORS.items()):
        yy = ly + i * 0.045
        svg.circle(lx + 0.01, yy, 0.008, fill=color, opacity=0.85)
        label = domain.replace('_', ' ').title()
        svg.text(lx + 0.035, yy, label, font_size=0.018, fill=color, opacity=0.75, text_anchor='start')

    return svg.to_svg()


# ═══════════════════════════════════════════════════════════════════════
# Poincaré Half-Plane (using v3 correct transform)
# ═══════════════════════════════════════════════════════════════════════

def render_halfplane():
    """Uses the notebook's transform: w = conj(i(1+conj(z))/(1-conj(z)))
    Maps to LOWER half-plane (y < 0), aligns with SVG y-down."""

    def hp_transform(x, y):
        z = complex(x, y)
        try:
            zc = z.conjugate()
            w = 1j * (1 + zc) / (1 - zc)
            w = w.conjugate()
            return (w.real, w.imag)
        except ZeroDivisionError:
            return (1e6, 1e6)

    # Bounding box
    pts = []
    for n in nodes:
        px, py = hp_transform(n['x'], n['y'])
        if abs(px) < 100 and abs(py) < 100:
            pts.append((px, py))

    xs = sorted(p[0] for p in pts)
    ys = sorted(p[1] for p in pts)
    clip = max(1, len(xs)//20)
    x_lo, x_hi = xs[clip], xs[-clip]
    y_lo, y_hi = ys[clip], ys[-clip]

    mx = (x_hi - x_lo) * 0.12
    my = (y_hi - y_lo) * 0.12
    vx = x_lo - mx
    vw = (x_hi - x_lo) + 2*mx
    vy = y_lo - my
    vh = (y_hi - y_lo) + 2*my
    if vh > vw * 0.55:
        extra = vh - vw * 0.55
        vy += extra * 0.3
        vh = vw * 0.55

    svg = SVGRenderer(vw, vh, viewbox=(vx, vy, vw, vh))

    svg.add_def('''
    <filter id="hp_glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="0.08" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <linearGradient id="hp_bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#02060c"/>
      <stop offset="100%" stop-color="#0b2845"/>
    </linearGradient>
    ''')

    svg.rect(vx, vy, vw, vh, fill='url(#hp_bg)')
    # Real axis at y=0 (math); in lower half-plane, points have y>0 in SVG coords actually... let me verify
    # conj flips imaginary sign. i(1+conj(z))/(1-conj(z)) for z=0 gives i, conj gives -i. So origin → (0, -1)
    # That's upper-half-plane (negative y = up). Hmm let me reconsider.
    # Actually: origin of disk → (0, -1) in this transform; boundary |z|=1 → real axis (y=0)
    # So nodes near disk center → y far from 0 (negative); near boundary → y near 0
    # In SVG y-down, negative y is UP. So we render with upper half plane layout.
    # Real axis is at y=0
    svg.line(vx, 0, vx+vw, 0, stroke='#4488cc', stroke_width=0.02, opacity=0.55)

    def hp_arc(x1, y1, x2, y2, **kw):
        if abs(x1 - x2) < 0.01:
            svg.line(x1, y1, x2, y2, **kw)
            return
        # Semicircle centered on real axis (y=0)
        # Center x: equidistant from both
        c = ((x1**2 + y1**2) - (x2**2 + y2**2)) / (2*(x1 - x2))
        r = math.hypot(x1 - c, y1)
        if r < 0.01 or r > 200:
            return
        a1 = math.atan2(y1, x1 - c)
        a2 = math.atan2(y2, x2 - c)
        sweep = 1 if a1 < a2 else 0
        d = f"M {x1},{y1} A {r},{r} 0 0,{sweep} {x2},{y2}"
        svg.path(d, **kw)

    # Horocycles as horizontal lines (y = const)
    y_range = [y_lo, y_hi]
    # Compute good horocycle y levels based on nodes
    ns = sorted(abs(p[1]) for p in pts)
    for frac in [0.25, 0.5, 0.75]:
        idx = int(len(ns) * frac)
        if idx < len(ns):
            y_level = -ns[idx]  # negative y (upper half)
            if vy <= y_level <= vy + vh:
                svg.line(vx, y_level, vx+vw, y_level,
                         stroke='#ffffff', stroke_width=0.005, opacity=0.08)

    # Dim edges
    svg.group_start(opacity='0.06')
    for e in dim_edges:
        x1, y1 = hp_transform(node_map[e['source']]['x'], node_map[e['source']]['y'])
        x2, y2 = hp_transform(node_map[e['target']]['x'], node_map[e['target']]['y'])
        if abs(y1) > 50 or abs(y2) > 50: continue
        c = domain_color(node_map[e['source']]['domain'])
        hp_arc(x1, y1, x2, y2, fill='none', stroke=c, stroke_width=0.02)
    svg.group_end()

    # Mid edges
    svg.group_start(opacity='0.22')
    for e in mid_edges:
        x1, y1 = hp_transform(node_map[e['source']]['x'], node_map[e['source']]['y'])
        x2, y2 = hp_transform(node_map[e['target']]['x'], node_map[e['target']]['y'])
        if abs(y1) > 50 or abs(y2) > 50: continue
        c = domain_color(node_map[e['source']]['domain'])
        hp_arc(x1, y1, x2, y2, fill='none', stroke=c, stroke_width=0.03)
    svg.group_end()

    # Top edges with glow
    svg.group_start(filter='url(#hp_glow)')
    for e in top_edges:
        x1, y1 = hp_transform(node_map[e['source']]['x'], node_map[e['source']]['y'])
        x2, y2 = hp_transform(node_map[e['target']]['x'], node_map[e['target']]['y'])
        if abs(y1) > 50 or abs(y2) > 50: continue
        c = domain_color(node_map[e['source']]['domain'])
        c_light = lighten(c, 0.3)
        hp_arc(x1, y1, x2, y2, fill='none', stroke=c, stroke_width=0.12, opacity=0.3)
        hp_arc(x1, y1, x2, y2, fill='none', stroke=c_light, stroke_width=0.04, opacity=0.75)
    svg.group_end()

    # Degree halos
    top_deg = sorted(nodes, key=lambda n: n['degree'], reverse=True)[:25]
    for n in top_deg:
        px, py = hp_transform(n['x'], n['y'])
        if abs(py) > 50 or abs(px) > 50: continue
        dist_axis = abs(py)
        scale = max(dist_axis * 0.12, 0.04)
        r = scale * math.log(1 + n['degree']) / max(math.log(1 + max_degree), 1)
        c = domain_color(n['domain'])
        svg.circle(px, py, max(r, 0.02), fill=c, opacity=0.12)
        svg.circle(px, py, max(r*0.6, 0.01), fill='none', stroke=c, stroke_width=0.015, opacity=0.3)

    # Node points (with data attrs)
    hp_positions = {}
    for n in sorted(nodes, key=lambda x: x['ppr']):
        px, py = hp_transform(n['x'], n['y'])
        if abs(py) > 50 or abs(px) > 50: continue
        hp_positions[n['id']] = (px, py)
        ppr_norm = (n['ppr'] - min_ppr) / (max_ppr - min_ppr) if max_ppr > min_ppr else 0.5
        ppr_norm = max(0, min(1, ppr_norm))
        dist_axis = abs(py)
        base_r = 0.03 + 0.1 * ppr_norm
        radius = base_r * max(min(dist_axis * 0.25, 1), 0.2)
        radius = max(radius, 0.02)
        c = domain_color(n['domain'])
        name_safe = n['name'].replace('"', '&quot;').replace('&', '&amp;')
        svg.raw(
            f'<circle cx="{px}" cy="{py}" r="{radius}" fill="{c}" opacity="0.9" '
            f'class="kg-node" data-id="{n["id"]}" data-name="{name_safe}" '
            f'data-domain="{n["domain"]}" data-ppr="{n["ppr"]:.6f}" '
            f'data-degree="{n["degree"]}" />'
        )
        svg.circle(px, py, radius*0.4, fill='white', opacity=0.6)

    # Labels for top PPR
    top20 = sorted(nodes, key=lambda n: n['ppr'], reverse=True)[:20]
    for n in top20:
        if n['id'] not in hp_positions: continue
        px, py = hp_positions[n['id']]
        name = n['name'].replace('_', ' ')
        if len(name) > 20:
            name = name[:18] + '..'
        ppr_norm = (n['ppr'] - min_ppr) / (max_ppr - min_ppr) if max_ppr > min_ppr else 0.5
        dist_axis = abs(py)
        fs = 0.08 + 0.06 * ppr_norm
        fs *= max(min(dist_axis * 0.25, 1), 0.25)
        c = lighten(domain_color(n['domain']), 0.4)
        svg.text(px, py + 0.12, name, font_size=max(fs, 0.04), fill=c, opacity=0.88)

    svg.text(vx + vw/2, vy + vh - 0.08,
             'POINCARÉ HALF-PLANE · Surfing Knowledge Graph',
             font_size=0.12, fill='#88ccff', opacity=0.75)

    # Legend
    lx = vx + 0.3
    ly = vy + 0.3
    for i, (domain, color) in enumerate(DOMAIN_COLORS.items()):
        yy = ly + i * 0.15
        svg.circle(lx + 0.05, yy, 0.035, fill=color, opacity=0.85)
        label = domain.replace('_', ' ').title()
        svg.text(lx + 0.15, yy, label, font_size=0.07, fill=color, opacity=0.75, text_anchor='start')

    return svg.to_svg()


if __name__ == '__main__':
    import os
    OUT = '/sessions/charming-exciting-cori/surfing_svgs'
    os.makedirs(OUT, exist_ok=True)

    print("Rendering Poincaré disk...")
    disk = render_disk()
    with open(f'{OUT}/surfing_disk.svg', 'w') as f:
        f.write(disk)
    print(f"  → {OUT}/surfing_disk.svg ({len(disk):,} bytes)")

    print("Rendering Poincaré half-plane...")
    hp = render_halfplane()
    with open(f'{OUT}/surfing_halfplane.svg', 'w') as f:
        f.write(hp)
    print(f"  → {OUT}/surfing_halfplane.svg ({len(hp):,} bytes)")

    print("Done.")
