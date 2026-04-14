#!/usr/bin/env python3
"""
Poincaré Disk & Half-Plane Knowledge Graph Visualization
=========================================================
Uses hyperbolic geometry primitives driven entirely by graph-computed values:
  - Point positions ← PPR rank (radial) × domain sector (angular)
  - Line geodesics ← graph edges
  - Hypercycle ribbons ← edge PPR (offset = geometric mean of endpoint PPRs)
  - Circle halos ← node degree (radius ∝ log(degree))
  - Horocycle arcs ← domain boundary sentinel nodes
  - Colors ← domain classification

Renders two views:
  1. Poincaré Disk (poincare_disk_hyp.svg)
  2. Poincaré Half-Plane via Transform.disk_to_half() (poincare_halfplane_hyp.svg)
"""

import json, math, random, colorsys
from hyperbolic_engine import (
    Point, Ideal, Line, Hypercycle, Circle, Horocycle, Transform,
    SVGRenderer, _clamp_to_disk
)

random.seed(42)

# ─── Load graph data ──────────────────────────────────────────────────
with open('/sessions/charming-exciting-cori/poincare_art_data.json') as f:
    data = json.load(f)

nodes = data['nodes']
edges = data['edges']

node_map = {n['id']: n for n in nodes}

# ─── Domain color palette ────────────────────────────────────────────
DOMAIN_COLORS = {
    'cognitive_neuro':   '#ff6b6b',  # coral red
    'cognitive_science': '#ff6b6b',
    'neuroscience':      '#ff8a8a',
    'ai_computing':      '#4ecdc4',  # teal
    'philosophy_mind':   '#a78bfa',  # lavender
    'philosophy':        '#a78bfa',
    'psychology':        '#f7b731',  # amber
    'linguistics':       '#45b7d1',  # sky blue
    'statistics_ml':     '#96f7a0',  # mint green
    'mathematics':       '#96f7a0',
    'biology_evolution': '#ff9ff3',  # pink
    'interdisciplinary': '#c8d6e5',  # silver blue
}

DOMAIN_SECTORS = {
    'cognitive_neuro':   0,
    'ai_computing':      45,
    'philosophy_mind':   90,
    'psychology':        135,
    'linguistics':       180,
    'statistics_ml':     225,
    'biology_evolution': 270,
    'interdisciplinary': 315,
}

def domain_color(d, alpha=1.0):
    return DOMAIN_COLORS.get(d, '#c8d6e5')

def darken(hex_color, factor=0.6):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255
    r, g, b = r*factor, g*factor, b*factor
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

def lighten(hex_color, factor=0.3):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255
    r = r + (1-r)*factor
    g = g + (1-g)*factor
    b = b + (1-b)*factor
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


# ─── Create hyperbolic Point objects from graph data ──────────────────
# Each node already has (x, y) in the Poincaré disk from our earlier pipeline
h_points = {}
for n in nodes:
    x, y = n['x'], n['y']
    # Ensure inside disk
    x, y = _clamp_to_disk(x, y, margin=0.02)
    h_points[n['id']] = Point(x, y)

# ─── Compute derived quantities ──────────────────────────────────────
ppr_values = [n['ppr'] for n in nodes]
max_ppr = max(ppr_values)
min_ppr = min(ppr_values)
max_degree = max(n['degree'] for n in nodes)

# ─── Build the edge index ────────────────────────────────────────────
edge_set = set()
valid_edges = []
for e in edges:
    s, t = e['source'], e['target']
    if s in node_map and t in node_map and s != t:
        key = (min(s, t), max(s, t))
        if key not in edge_set:
            edge_set.add(key)
            valid_edges.append(e)

print(f"Nodes: {len(nodes)}, Valid edges: {len(valid_edges)}")

# ─── Identify domain boundary sentinels (for horocycles) ─────────────
# Pick the outermost node in each domain sector
domain_sentinels = {}
for n in nodes:
    d = n['domain']
    r = math.hypot(n['x'], n['y'])
    if d not in domain_sentinels or r > domain_sentinels[d][1]:
        domain_sentinels[d] = (n['id'], r)

# ─── Categorize edges by importance ──────────────────────────────────
def edge_importance(e):
    s_ppr = node_map[e['source']]['ppr']
    t_ppr = node_map[e['target']]['ppr']
    return math.sqrt(s_ppr * t_ppr)

valid_edges.sort(key=edge_importance, reverse=True)
top_edges = valid_edges[:40]      # Hypercycle ribbons
mid_edges = valid_edges[40:150]   # Geodesic lines
dim_edges = valid_edges[150:]     # Faint background geodesics

# ═══════════════════════════════════════════════════════════════════════
#  POINCARÉ DISK VIEW
# ═══════════════════════════════════════════════════════════════════════

def render_disk():
    svg = SVGRenderer(2.2, 2.2, origin='center')

    # ── Defs: filters and gradients ──
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
      <stop offset="0%" stop-color="#0a0a1a"/>
      <stop offset="70%" stop-color="#0d0d2b"/>
      <stop offset="100%" stop-color="#1a1a3e"/>
    </radialGradient>
    <radialGradient id="boundary_glow" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse">
      <stop offset="85%" stop-color="transparent"/>
      <stop offset="95%" stop-color="rgba(100,100,200,0.15)"/>
      <stop offset="100%" stop-color="rgba(150,150,255,0.3)"/>
    </radialGradient>
    ''')

    # Domain sector gradient wedges
    for domain, base_angle in DOMAIN_SECTORS.items():
        color = DOMAIN_COLORS.get(domain, '#888')
        svg.add_def(f'''
    <radialGradient id="sector_{domain}" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="transparent"/>
      <stop offset="60%" stop-color="{color}" stop-opacity="0.02"/>
      <stop offset="100%" stop-color="{color}" stop-opacity="0.08"/>
    </radialGradient>''')

    # ── Background ──
    svg.circle(0, 0, 1.05, fill='#050510')
    svg.circle(0, 0, 1.0, fill='url(#diskbg)')
    svg.circle(0, 0, 1.0, fill='url(#boundary_glow)')

    # ── Domain sector washes ──
    for domain, base_angle in DOMAIN_SECTORS.items():
        color = DOMAIN_COLORS.get(domain, '#888')
        a1 = math.radians(base_angle - 22.5)
        a2 = math.radians(base_angle + 22.5)
        # Draw a wedge
        x1 = math.cos(a1)
        y1 = math.sin(a1)
        x2 = math.cos(a2)
        y2 = math.sin(a2)
        d = f"M 0,0 L {x1},{y1} A 1,1 0 0,1 {x2},{y2} Z"
        svg.path(d, fill=color, opacity=0.06)

    # ── PPR isocline rings ──
    for r_val in [0.2, 0.4, 0.6, 0.8]:
        svg.circle(0, 0, r_val, fill='none', stroke='#ffffff', stroke_width=0.002, opacity=0.12)

    # ── Horocycles for domain boundary sentinels ──
    for domain, (nid, radius) in domain_sentinels.items():
        p = h_points[nid]
        color = domain_color(domain)
        horo = Horocycle.from_closest_point(p, surround_origin=False)
        svg.draw_horocycle(horo, hwidth=0.008, fill=color, opacity=0.35)
        # Second horocycle (surrounding) for high-PPR domains
        if node_map[nid]['ppr'] > 0.003:
            horo2 = Horocycle.from_closest_point(p, surround_origin=True)
            svg.draw_horocycle(horo2, hwidth=0.005, fill=color, opacity=0.15)

    # ── Dim background edges (geodesic lines) ──
    svg.group_start(opacity='0.08')
    for e in dim_edges:
        s = node_map[e['source']]
        t = node_map[e['target']]
        p1 = h_points[e['source']]
        p2 = h_points[e['target']]
        line = Line.from_points(*p1, *p2, segment=True)
        # Color: blend of source and target domains
        c = domain_color(s['domain'])
        svg.draw_geodesic(line, hwidth=0.003, fill=c, stroke=c, stroke_width=0.003)
    svg.group_end()

    # ── Mid-tier edges as geodesic lines ──
    svg.group_start(opacity='0.25')
    for e in mid_edges:
        s = node_map[e['source']]
        t = node_map[e['target']]
        p1 = h_points[e['source']]
        p2 = h_points[e['target']]
        line = Line.from_points(*p1, *p2, segment=True)
        c = domain_color(s['domain'])
        svg.draw_geodesic(line, hwidth=0.005, fill=c, stroke=c, stroke_width=0.005)
    svg.group_end()

    # ── Top edges as Hypercycle ribbons ──
    svg.group_start(filter='url(#softglow)')
    for i, e in enumerate(top_edges):
        s = node_map[e['source']]
        t = node_map[e['target']]
        p1 = h_points[e['source']]
        p2 = h_points[e['target']]
        line = Line.from_points(*p1, *p2, segment=True)
        # Offset proportional to geometric mean of PPR values
        geo_ppr = math.sqrt(s['ppr'] * t['ppr'])
        offset = 0.08 + 0.4 * (geo_ppr / max_ppr)
        c = domain_color(s['domain'])
        c_light = lighten(c, 0.3)
        # Wide glow ribbon
        hc1 = Hypercycle.from_hypercycle_offset(line, offset)
        svg.draw_hypercycle(hc1, hwidth=0.025, fill=c, opacity=0.3)
        # Thin bright core
        hc2 = Hypercycle.from_hypercycle_offset(line, offset * 0.5)
        svg.draw_hypercycle(hc2, hwidth=0.008, fill=c_light, opacity=0.7)
        # Mirror ribbon (negative offset)
        hc3 = Hypercycle.from_hypercycle_offset(line, -offset)
        svg.draw_hypercycle(hc3, hwidth=0.018, fill=c, opacity=0.2)
        # Central geodesic
        svg.draw_geodesic(line, hwidth=0.004, fill='white', stroke='white', stroke_width=0.004, opacity=0.5)
    svg.group_end()

    # ── Hyperbolic Circles (degree halos) for top nodes ──
    top_nodes = sorted(nodes, key=lambda n: n['degree'], reverse=True)[:30]
    for n in top_nodes:
        p = h_points[n['id']]
        # Hyperbolic radius proportional to log(degree)
        h_r = 0.05 + 0.15 * math.log(1 + n['degree']) / math.log(1 + max_degree)
        c = domain_color(n['domain'])
        circ = Circle.from_center_radius(p, h_r)
        svg.draw_circle(circ, fill=c, opacity=0.12)
        # Inner ring
        circ2 = Circle.from_center_radius(p, h_r * 0.6)
        svg.draw_circle(circ2, fill='none', stroke=c, stroke_width=0.004, opacity=0.3)

    # ── Node points ──
    # Size by PPR, color by domain
    for n in sorted(nodes, key=lambda n: n['ppr']):
        p = h_points[n['id']]
        ppr_norm = (n['ppr'] - min_ppr) / (max_ppr - min_ppr) if max_ppr > min_ppr else 0.5
        radius = 0.008 + 0.025 * ppr_norm
        c = domain_color(n['domain'])
        svg.draw_point(p, radius=radius, fill=c, opacity=0.9)
        # Bright core
        svg.draw_point(p, radius=radius*0.4, fill='white', opacity=0.6)

    # ── Labels for top-20 nodes ──
    top20 = sorted(nodes, key=lambda n: n['ppr'], reverse=True)[:20]
    for n in top20:
        p = h_points[n['id']]
        name = n['name'].replace('_', ' ')
        if len(name) > 18:
            name = name[:16] + '..'
        ppr_norm = (n['ppr'] - min_ppr) / (max_ppr - min_ppr)
        fs = 0.022 + 0.012 * ppr_norm
        c = lighten(domain_color(n['domain']), 0.4)
        svg.text(p.x, p.y - 0.04, name, font_size=fs, fill=c, opacity=0.85)

    # ── Boundary circle ──
    svg.circle(0, 0, 1.0, fill='none', stroke='#4444aa', stroke_width=0.006, opacity=0.5)

    # ── Title ──
    svg.text(0, -1.07, 'POINCARÉ DISK · Knowledge Hyperbolic Embedding',
             font_size=0.035, fill='#8888cc', opacity=0.7)

    # ── Legend ──
    lx, ly = -1.05, 0.75
    for i, (domain, color) in enumerate(DOMAIN_COLORS.items()):
        yy = ly + i * 0.045
        svg.circle(lx + 0.01, yy, 0.008, fill=color, opacity=0.8)
        label = domain.replace('_', ' ').title()
        svg.text(lx + 0.035, yy, label, font_size=0.018, fill=color, opacity=0.7, text_anchor='start')

    return svg.to_svg()


# ═══════════════════════════════════════════════════════════════════════
#  POINCARÉ HALF-PLANE VIEW
# ═══════════════════════════════════════════════════════════════════════

def render_halfplane():
    # Transform: disk → half-plane
    trans = Transform.merge(
        Transform.rotation(deg=90),
        Transform.disk_to_half(),
    )

    # Helper: apply transform and flip y for SVG (SVG y-down, math y-up)
    def hp_transform(z):
        w = trans.apply(z)
        return (w.real, -w.imag)  # Negate y so upper half-plane renders upward

    # Compute bounding box
    test_pts = []
    for n in nodes:
        z = complex(n['x'], n['y'])
        px, py = hp_transform(z)
        if abs(py) < 50 and abs(px) < 50:
            test_pts.append((px, py))

    if not test_pts:
        vx, vy, vw, vh = -5, -6, 10, 6.5
    else:
        xs = [p[0] for p in test_pts]
        ys = [p[1] for p in test_pts]
        # Clip outliers (5%) for tighter framing
        xs_s = sorted(xs)
        ys_s = sorted(ys)
        clip = max(1, len(xs_s) // 20)
        x_lo, x_hi = xs_s[clip], xs_s[-clip]
        y_lo, y_hi = ys_s[clip], ys_s[-clip]
        margin_x = (x_hi - x_lo) * 0.12
        margin_y = (y_hi - y_lo) * 0.12
        vx = x_lo - margin_x
        vw = (x_hi - x_lo) + 2 * margin_x
        vy = y_lo - margin_y
        vh = (y_hi - y_lo) + 2 * margin_y
        # Ensure wider-than-tall aspect for a nice panoramic look
        if vh > vw * 0.6:
            extra = vh - vw * 0.6
            vy += extra * 0.3  # Trim more from top (less interesting)
            vh = vw * 0.6

    svg = SVGRenderer(vw, vh, viewbox=(vx, vy, vw, vh))

    # ── Defs ──
    svg.add_def('''
    <filter id="hp_glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="0.08" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <linearGradient id="hp_bg" x1="0" y1="1" x2="0" y2="0">
      <stop offset="0%" stop-color="#050510"/>
      <stop offset="100%" stop-color="#0d0d2b"/>
    </linearGradient>
    ''')

    # ── Helper: draw a half-plane geodesic arc between two SVG points ──
    def hp_arc(svg_obj, x1, y1, x2, y2, **kw):
        """Draw a half-plane geodesic (semicircle on real axis or vertical line)."""
        # Real axis is at y=0 in SVG after our flip
        # Geodesics in half-plane: semicircles centered on real axis
        real_axis_y = 0  # After flip, real axis is at SVG y=0
        if abs(x1 - x2) < 0.01:
            svg_obj.line(x1, y1, x2, y2, **kw)
        else:
            # Find circle center on the real axis: equidistant to both points
            # Using the original (unflipped) half-plane coords to get the geodesic right:
            # The points in math coords are (x1, -y1) and (x2, -y2) [undo SVG flip]
            my1, my2 = -y1, -y2  # Math y coords (positive = up)
            dx = x2 - x1
            if abs(dx) > 0.01:
                cx_hp = ((x1**2 + my1**2) - (x2**2 + my2**2)) / (2*dx)
            else:
                cx_hp = (x1 + x2) / 2
            r_hp = math.hypot(x1 - cx_hp, my1)
            if 0.01 < r_hp < 50:
                # Arc in SVG coords (y flipped): sweep direction flips
                sweep = '0' if x1 < x2 else '1'
                d = f"M {x1},{y1} A {r_hp},{r_hp} 0 0,{sweep} {x2},{y2}"
                svg_obj.path(d, **kw)

    # ── Background ──
    svg.rect(vx, vy, vw, vh, fill='url(#hp_bg)')
    # Real axis (at SVG y=0, which is math y=0)
    svg.line(vx, 0, vx + vw, 0, stroke='#4444aa', stroke_width=0.02, opacity=0.5)

    # ── Dim background edges ──
    svg.group_start(opacity='0.06')
    for e in dim_edges:
        s, t = e['source'], e['target']
        z1, z2 = complex(h_points[s].x, h_points[s].y), complex(h_points[t].x, h_points[t].y)
        x1, y1 = hp_transform(z1)
        x2, y2 = hp_transform(z2)
        if abs(y1) > 30 or abs(y2) > 30 or abs(x1) > 30 or abs(x2) > 30:
            continue
        c = domain_color(node_map[s]['domain'])
        hp_arc(svg, x1, y1, x2, y2, fill='none', stroke=c, stroke_width=0.02)
    svg.group_end()

    # ── Mid-tier edges ──
    svg.group_start(opacity='0.2')
    for e in mid_edges:
        s, t = e['source'], e['target']
        z1, z2 = complex(h_points[s].x, h_points[s].y), complex(h_points[t].x, h_points[t].y)
        x1, y1 = hp_transform(z1)
        x2, y2 = hp_transform(z2)
        if abs(y1) > 30 or abs(y2) > 30:
            continue
        c = domain_color(node_map[s]['domain'])
        hp_arc(svg, x1, y1, x2, y2, fill='none', stroke=c, stroke_width=0.03)
    svg.group_end()

    # ── Top edges as hypercycle ribbons ──
    svg.group_start(filter='url(#hp_glow)')
    for e in top_edges:
        s, t = e['source'], e['target']
        z1, z2 = complex(h_points[s].x, h_points[s].y), complex(h_points[t].x, h_points[t].y)
        x1, y1 = hp_transform(z1)
        x2, y2 = hp_transform(z2)
        if abs(y1) > 30 or abs(y2) > 30:
            continue
        c = domain_color(node_map[s]['domain'])
        c_light = lighten(c, 0.3)
        # Wide glow
        hp_arc(svg, x1, y1, x2, y2, fill='none', stroke=c, stroke_width=0.12, opacity=0.3)
        # Bright core
        hp_arc(svg, x1, y1, x2, y2, fill='none', stroke=c_light, stroke_width=0.04, opacity=0.7)
    svg.group_end()

    # ── Degree halos ──
    top_deg_nodes = sorted(nodes, key=lambda n: n['degree'], reverse=True)[:30]
    for n in top_deg_nodes:
        z = complex(h_points[n['id']].x, h_points[n['id']].y)
        px, py = hp_transform(z)
        if abs(py) > 30 or abs(px) > 30:
            continue
        # Scale radius with distance from real axis (conformal factor)
        math_y = abs(py)  # Distance from axis in SVG = distance in math
        scale = max(math_y * 0.12, 0.04)
        r = scale * math.log(1 + n['degree']) / math.log(1 + max_degree)
        c = domain_color(n['domain'])
        svg.circle(px, py, max(r, 0.02), fill=c, opacity=0.12)
        svg.circle(px, py, max(r*0.6, 0.01), fill='none', stroke=c, stroke_width=0.015, opacity=0.3)

    # ── Horocycle reference lines (horizontal = horocycles in half-plane) ──
    for math_y in [0.5, 1.0, 2.0, 4.0]:
        svg_y = -math_y  # Flip for SVG
        if vy <= svg_y <= vy + vh:
            svg.line(vx, svg_y, vx + vw, svg_y, stroke='#ffffff', stroke_width=0.008, opacity=0.08)

    # ── Node points ──
    hp_positions = {}
    for n in sorted(nodes, key=lambda n: n['ppr']):
        z = complex(h_points[n['id']].x, h_points[n['id']].y)
        px, py = hp_transform(z)
        if abs(py) > 30 or abs(px) > 30:
            continue
        hp_positions[n['id']] = (px, py)
        ppr_norm = (n['ppr'] - min_ppr) / (max_ppr - min_ppr) if max_ppr > min_ppr else 0.5
        math_y = abs(py)
        base_r = 0.03 + 0.1 * ppr_norm
        radius = base_r * max(min(math_y * 0.25, 1), 0.15)
        radius = max(radius, 0.02)
        c = domain_color(n['domain'])
        svg.circle(px, py, radius, fill=c, opacity=0.9)
        svg.circle(px, py, radius*0.4, fill='white', opacity=0.6)

    # ── Labels for top-20 ──
    top20 = sorted(nodes, key=lambda n: n['ppr'], reverse=True)[:20]
    for n in top20:
        if n['id'] not in hp_positions:
            continue
        px, py = hp_positions[n['id']]
        name = n['name'].replace('_', ' ')
        if len(name) > 18:
            name = name[:16] + '..'
        ppr_norm = (n['ppr'] - min_ppr) / (max_ppr - min_ppr)
        math_y = abs(py)
        fs = 0.08 + 0.05 * ppr_norm
        fs *= max(min(math_y * 0.25, 1), 0.25)
        c = lighten(domain_color(n['domain']), 0.4)
        svg.text(px, py + 0.12, name, font_size=max(fs, 0.04), fill=c, opacity=0.85)

    # ── Title ──
    svg.text((vx + vw/2), vy + vh - 0.1,
             'POINCARÉ HALF-PLANE · Knowledge Hyperbolic Embedding',
             font_size=0.12, fill='#8888cc', opacity=0.7)

    # ── Legend ──
    lx = vx + 0.2
    ly = vy + 0.3
    for i, (domain, color) in enumerate(DOMAIN_COLORS.items()):
        yy = ly + i * 0.14
        svg.circle(lx + 0.05, yy, 0.03, fill=color, opacity=0.8)
        label = domain.replace('_', ' ').title()
        svg.text(lx + 0.15, yy, label, font_size=0.06, fill=color, opacity=0.7, text_anchor='start')

    return svg.to_svg()


# ═══════════════════════════════════════════════════════════════════════
#  RENDER
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Rendering Poincaré disk...")
    disk_svg = render_disk()
    disk_path = '/sessions/charming-exciting-cori/mnt/rag/poincare_disk_hyp.svg'
    with open(disk_path, 'w') as f:
        f.write(disk_svg)
    print(f"  → {disk_path} ({len(disk_svg):,} bytes)")

    print("Rendering Poincaré half-plane...")
    hp_svg = render_halfplane()
    hp_path = '/sessions/charming-exciting-cori/mnt/rag/poincare_halfplane_hyp.svg'
    with open(hp_path, 'w') as f:
        f.write(hp_svg)
    print(f"  → {hp_path} ({len(hp_svg):,} bytes)")

    print("\nDone! Both views rendered.")
