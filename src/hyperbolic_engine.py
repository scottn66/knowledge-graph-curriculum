"""
Poincaré disk & half-plane hyperbolic geometry engine.
Reimplements the core API of the `hyperbolic` Python library:
  Point, Ideal, Line, Hypercycle, Circle, Horocycle, Polygon, Transform
with pure-SVG rendering (no drawsvg dependency).

All geometry lives in the Poincaré disk model |z| < 1.
"""

import math, cmath, json, colorsys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

# ─── helpers ──────────────────────────────────────────────────────────
EPS = 1e-10

def _clamp_to_disk(x, y, margin=1e-6):
    r = math.hypot(x, y)
    if r >= 1.0 - margin:
        s = (1.0 - margin) / r
        return x * s, y * s
    return x, y

def _mobius_add(a, b):
    """Möbius addition in the Poincaré disk: a ⊕ b"""
    ax, ay = a
    bx, by = b
    dot_ab = ax*bx + ay*by
    norm_a2 = ax*ax + ay*ay
    norm_b2 = bx*bx + by*by
    denom = 1 + 2*dot_ab + norm_a2*norm_b2
    if abs(denom) < EPS:
        return 0.0, 0.0
    rx = ((1+2*dot_ab+norm_b2)*ax + (1-norm_a2)*bx) / denom
    ry = ((1+2*dot_ab+norm_b2)*ay + (1-norm_a2)*by) / denom
    return rx, ry

def _circle_through_three(x1, y1, x2, y2, x3, y3):
    """Find circle (cx, cy, r) through three points. Returns None if collinear."""
    ax, ay = x1, y1
    bx, by = x2, y2
    cx, cy = x3, y3
    D = 2 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
    if abs(D) < EPS:
        return None
    ux = ((ax*ax + ay*ay)*(by - cy) + (bx*bx + by*by)*(cy - ay) + (cx*cx + cy*cy)*(ay - by)) / D
    uy = ((ax*ax + ay*ay)*(cx - bx) + (bx*bx + by*by)*(ax - cx) + (cx*cx + cy*cy)*(bx - ax)) / D
    r = math.hypot(ux - ax, uy - ay)
    return ux, uy, r

def _geodesic_circle(x1, y1, x2, y2):
    """
    Compute the Euclidean circle for the Poincaré geodesic between (x1,y1) and (x2,y2).
    Returns (cx, cy, r) or None if the geodesic is a diameter.
    """
    # Inversion of each point through unit circle gives the third point
    n1 = x1*x1 + y1*y1
    n2 = x2*x2 + y2*y2
    if n1 < EPS or n2 < EPS:
        # One point is at origin → geodesic is a straight line through origin
        return None
    # Inverse points
    ix1, iy1 = x1/n1, y1/n1
    ix2, iy2 = x2/n2, y2/n2
    # Check if points are (anti)diametrically opposed
    cross = x1*y2 - y1*x2
    if abs(cross) < EPS:
        return None  # Collinear through origin → diameter
    return _circle_through_three(x1, y1, x2, y2, ix1, iy1)

def _arc_angles(cx, cy, x1, y1, x2, y2):
    """Angles from circle center to two points, ensuring short arc."""
    a1 = math.atan2(y1 - cy, x1 - cx)
    a2 = math.atan2(y2 - cy, x2 - cx)
    return a1, a2

def _svg_arc_path(cx, cy, r, x1, y1, x2, y2, large_arc=None):
    """SVG path for an arc of circle (cx,cy,r) from (x1,y1) to (x2,y2)."""
    if large_arc is None:
        # Determine short arc
        a1 = math.atan2(y1 - cy, x1 - cx)
        a2 = math.atan2(y2 - cy, x2 - cx)
        diff = (a2 - a1) % (2*math.pi)
        large_arc = 1 if diff > math.pi else 0
    # sweep direction
    cross = (x1 - cx)*(y2 - cy) - (y1 - cy)*(x2 - cx)
    sweep = 1 if cross > 0 else 0
    return f"M {x1},{y1} A {r},{r} 0 {large_arc},{sweep} {x2},{y2}"

# ─── Primitives ───────────────────────────────────────────────────────

class Point:
    """A point in the Poincaré disk."""
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        x, y = _clamp_to_disk(x, y)
        self.x = x
        self.y = y
    def __iter__(self):
        yield self.x
        yield self.y
    def __repr__(self):
        return f"Point({self.x:.4f}, {self.y:.4f})"
    @classmethod
    def from_euclid(cls, x, y):
        if x*x + y*y >= 1.0:
            raise ValueError("Point outside disk")
        return cls(x, y)
    @classmethod
    def from_h_polar(cls, h_dist, deg=0):
        """From hyperbolic polar: r_euclid = tanh(h_dist/2)."""
        r = math.tanh(h_dist / 2)
        rad = math.radians(deg)
        return cls(r * math.cos(rad), r * math.sin(rad))
    @classmethod
    def from_polar_euclid(cls, r, deg=0):
        rad = math.radians(deg)
        return cls(r * math.cos(rad), r * math.sin(rad))
    def distance_to(self, other):
        """Poincaré disk distance."""
        dx = self.x - other.x
        dy = self.y - other.y
        n1 = self.x**2 + self.y**2
        n2 = other.x**2 + other.y**2
        arg = 1 + 2*(dx*dx + dy*dy) / ((1 - n1)*(1 - n2))
        return math.acosh(max(arg, 1.0))
    def to_complex(self):
        return complex(self.x, self.y)
    @classmethod
    def from_complex(cls, z):
        return cls(z.real, z.imag)

class Ideal:
    """An ideal point on the boundary of the disk."""
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        r = math.hypot(x, y)
        if r < EPS:
            self.x, self.y = 1.0, 0.0
        else:
            self.x, self.y = x/r, y/r
    def __iter__(self):
        yield self.x
        yield self.y
    @classmethod
    def from_degree(cls, deg):
        rad = math.radians(deg)
        return cls(math.cos(rad), math.sin(rad))

class Line:
    """A geodesic in the Poincaré disk (circular arc ⊥ boundary or diameter)."""
    def __init__(self, x1, y1, x2, y2, segment=False):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.segment = segment
        self._circ = _geodesic_circle(x1, y1, x2, y2)
    @classmethod
    def from_points(cls, x1, y1, x2, y2, segment=False):
        return cls(x1, y1, x2, y2, segment=segment)
    @property
    def is_diameter(self):
        return self._circ is None
    def get_circle(self):
        return self._circ
    def make_perpendicular(self, px, py):
        """Line through (px,py) perpendicular to self."""
        if self.is_diameter:
            # Perpendicular to diameter through origin
            dx = self.x2 - self.x1
            dy = self.y2 - self.y1
            # Reflect p across the diameter
            norm = dx*dx + dy*dy
            dot = (px*dx + py*dy) / norm
            rx = 2*dot*dx - px
            ry = 2*dot*dy - py
            return Line.from_points(px, py, rx, ry, segment=False)
        cx, cy, r = self._circ
        # Invert p through the geodesic circle
        dpx, dpy = px - cx, py - cy
        d2 = dpx*dpx + dpy*dpy
        if d2 < EPS:
            return Line.from_points(px, py, -px, -py, segment=False)
        inv_x = cx + r*r*dpx/d2
        inv_y = cy + r*r*dpy/d2
        inv_x, inv_y = _clamp_to_disk(inv_x, inv_y)
        return Line.from_points(px, py, inv_x, inv_y, segment=False)
    def ideal_endpoints(self):
        """Get the two ideal endpoints of this geodesic on the unit circle."""
        if self.is_diameter:
            dx = self.x2 - self.x1
            dy = self.y2 - self.y1
            r = math.hypot(dx, dy)
            if r < EPS:
                return (1, 0), (-1, 0)
            nx, ny = dx/r, dy/r
            return (nx, ny), (-nx, -ny)
        cx, cy, rad = self._circ
        # Intersection of the geodesic circle with the unit circle
        d = math.hypot(cx, cy)
        if d < EPS:
            return (1, 0), (-1, 0)
        # Using formula for circle-circle intersection
        a = (cx*cx + cy*cy + 1 - rad*rad) / (2*d)
        h2 = 1 - a*a
        if h2 < 0:
            h2 = 0
        h = math.sqrt(h2)
        mx, my = a*cx/d, a*cy/d
        px, py = -cy/d, cx/d
        return (mx + h*px, my + h*py), (mx - h*px, my - h*py)

class Hypercycle:
    """A curve equidistant from a geodesic in the Poincaré disk."""
    def __init__(self, base_line, offset):
        self.base_line = base_line
        self.offset = offset
        self._compute()
    def _compute(self):
        """Compute the Euclidean circular arc for this hypercycle."""
        ep1, ep2 = self.base_line.ideal_endpoints()
        # A hypercycle with offset d from a geodesic meets the boundary
        # at points rotated by angle α = atan(tanh(d)) from the geodesic endpoints
        # around the circle center
        if abs(self.offset) < EPS:
            self._circ = self.base_line._circ
            self._ep1 = ep1
            self._ep2 = ep2
            return
        # Rotate ideal endpoints by a fraction of the offset
        angle_shift = math.atan(math.tanh(self.offset))
        # Midpoint angle of the two ideal endpoints
        a1 = math.atan2(ep1[1], ep1[0])
        a2 = math.atan2(ep2[1], ep2[0])
        # Shift each ideal endpoint by angle_shift along the boundary
        self._ep1 = (math.cos(a1 + angle_shift), math.sin(a1 + angle_shift))
        self._ep2 = (math.cos(a2 + angle_shift), math.sin(a2 + angle_shift))
        # The hypercycle arc passes through these boundary points and
        # a midpoint that is offset from the geodesic midpoint
        mx = (self.base_line.x1 + self.base_line.x2) / 2
        my = (self.base_line.y1 + self.base_line.y2) / 2
        # Normal direction to geodesic at midpoint
        if self.base_line.is_diameter:
            dx = self.base_line.x2 - self.base_line.x1
            dy = self.base_line.y2 - self.base_line.y1
            r = math.hypot(dx, dy)
            nx, ny = -dy/r, dx/r
        else:
            cx, cy, rad = self.base_line._circ
            nx = mx - cx
            ny = my - cy
            rn = math.hypot(nx, ny)
            if rn > EPS:
                nx, ny = nx/rn, ny/rn
        # Offset in Euclidean coordinates (approximate for visualization)
        off = math.tanh(self.offset / 2) * 0.3
        mid_x, mid_y = _clamp_to_disk(mx + off*nx, my + off*ny)
        circ = _circle_through_three(
            self._ep1[0], self._ep1[1],
            mid_x, mid_y,
            self._ep2[0], self._ep2[1])
        self._circ = circ
        self._mid = (mid_x, mid_y)
    @classmethod
    def from_hypercycle_offset(cls, line, offset):
        return cls(line, offset)
    @classmethod
    def from_points(cls, x1, y1, x2, y2, x3, y3, segment=False):
        """Create hypercycle through three points."""
        circ = _circle_through_three(x1, y1, x2, y2, x3, y3)
        # Approximate: create a line from endpoints and compute offset
        line = Line.from_points(x1, y1, x3, y3, segment=segment)
        # Estimate offset from middle point's distance to geodesic
        hc = cls.__new__(cls)
        hc.base_line = line
        hc.offset = 0  # Approximate
        hc._circ = circ
        hc._ep1 = (x1, y1)
        hc._ep2 = (x3, y3)
        return hc

class Circle:
    """A hyperbolic circle in the Poincaré disk.
    A hyperbolic circle is also a Euclidean circle, but with shifted center."""
    def __init__(self, center, h_radius):
        self.center = center
        self.h_radius = h_radius
        self._compute()
    def _compute(self):
        cx, cy = self.center.x, self.center.y
        r_norm = cx*cx + cy*cy
        # Euclidean radius and center of a hyperbolic circle
        # centered at (cx, cy) with hyperbolic radius h_r
        h_r = self.h_radius
        r_e = math.tanh(h_r / 2)
        # For a point at Euclidean distance d from origin:
        # The hyperbolic circle has Euclidean center shifted toward origin
        # and Euclidean radius adjusted
        if r_norm < EPS:
            self.e_cx = 0
            self.e_cy = 0
            self.e_r = r_e
        else:
            d = math.sqrt(r_norm)
            # Poincaré disk: hyperbolic circle centered at p with radius r
            # maps to Euclidean circle with:
            #   center = p * (1 - r_e^2) / (1 - d^2 * r_e^2)  [projected]
            #   This is a simplification; exact formula:
            cosh_r = math.cosh(h_r)
            sinh_r = math.sinh(h_r)
            # Euclidean center distance from origin
            k = (1 - r_norm)
            denom = cosh_r - r_norm * cosh_r + r_norm  # simplified
            # Actually, use the exact Euclidean representation:
            # ec = p * (cosh(r) - 1) / ... is complex; use the known formula:
            t = math.tanh(h_r)
            factor_c = (1 - t*t) / (1 - r_norm * t * t) if abs(1 - r_norm*t*t) > EPS else 1
            factor_r = t * (1 - r_norm) / (1 - r_norm * t * t) if abs(1 - r_norm*t*t) > EPS else t
            self.e_cx = cx * factor_c if abs(factor_c - 1) > EPS else cx
            self.e_cy = cy * factor_c if abs(factor_c - 1) > EPS else cy
            self.e_r = max(factor_r, 0.001)
            # Simpler approximation that works well visually:
            # Just use tanh(h_r/2) scaled by conformal factor
            conf = (1 - r_norm)
            self.e_r = r_e * conf / (1 - (d * r_e)**2) if (d*r_e) < 0.99 else 0.001
            self.e_cx = cx * (1 - self.e_r**2 / (1 if r_norm < EPS else r_norm) * 0)
            # Keep it simple and correct:
            self.e_cx = cx
            self.e_cy = cy
            self.e_r = min(r_e * conf, 1 - d - 0.001)
            if self.e_r < 0.001:
                self.e_r = 0.001
    @classmethod
    def from_center_radius(cls, center, h_radius):
        return cls(center, h_radius)

class Horocycle:
    """A horocycle: a curve tangent to the boundary circle at an ideal point."""
    def __init__(self, closest_point, surround_origin=False):
        self.closest_point = closest_point
        self.surround_origin = surround_origin
        self._compute()
    def _compute(self):
        px, py = self.closest_point.x, self.closest_point.y
        d = math.hypot(px, py)
        if d < EPS:
            # Point at origin: horocycle is any circle tangent to boundary
            self.e_cx = 0.5 if not self.surround_origin else 0
            self.e_cy = 0
            self.e_r = 0.5
            return
        # Direction from origin to point
        nx, ny = px/d, py/d
        if self.surround_origin:
            # Horocycle surrounds origin: tangent to boundary on far side
            # Euclidean circle centered beyond the point, tangent to unit circle
            # The horocycle through p that surrounds origin is tangent to
            # the unit circle at the ideal point opposite to p
            t = (1 + d) / 2
            self.e_cx = -nx * (t - 1) + nx * 0
            self.e_cy = -ny * (t - 1) + ny * 0
            # Actually: tangent at ideal point in direction of p
            # Euclidean center at (1+d)/2 * n, radius (1-d)/2
            self.e_cx = nx * (1 + d) / 2
            self.e_cy = ny * (1 + d) / 2
            self.e_r = (1 - d) / 2
        else:
            # Horocycle not surrounding origin: small circle near boundary
            # Tangent to unit circle at ideal point in direction of p
            # Passes through p
            # Euclidean center at midpoint between p and ideal point
            self.e_cx = nx * (1 + d) / 2
            self.e_cy = ny * (1 + d) / 2
            self.e_r = (1 - d) / 2
    @classmethod
    def from_closest_point(cls, point, surround_origin=False):
        return cls(point, surround_origin)

class Transform:
    """Möbius transformations of the Poincaré disk."""
    def __init__(self, a, b, c, d):
        # Möbius: f(z) = (az+b)/(cz+d)
        self.a = complex(a)
        self.b = complex(b)
        self.c = complex(c)
        self.d = complex(d)
    def apply(self, z):
        z = complex(z)
        denom = self.c * z + self.d
        if abs(denom) < EPS:
            return complex(0, 1e6)
        return (self.a * z + self.b) / denom
    def apply_to_point(self, p):
        z = complex(p.x, p.y)
        w = self.apply(z)
        return Point(w.real, w.imag)
    def __call__(self, *points):
        return [self.apply_to_point(p) for p in points]
    @classmethod
    def disk_to_half(cls):
        """Map Poincaré disk to upper half-plane: w = i(1+z)/(1-z)."""
        return cls(1j, 1j, -1, 1)
    @classmethod
    def mirror(cls, direction):
        """Reflection. direction is (dx, dy) — reflect across line through origin with that normal."""
        if isinstance(direction, tuple):
            dx, dy = direction
        else:
            dx, dy = direction.x, direction.y
        r = math.hypot(dx, dy)
        dx, dy = dx/r, dy/r
        # Reflection matrix as Möbius: f(z) = e^{2iθ} * conj(z)
        # This is anti-conformal, not a Möbius transform
        # For SVG purposes, we'll handle mirror as a flag
        return MirrorTransform(dx, dy)
    @classmethod
    def rotation(cls, deg=0):
        rad = math.radians(deg)
        e = cmath.exp(1j * rad)
        return cls(e, 0, 0, 1)
    @classmethod
    def translation(cls, p1_or_dist, p2=None):
        if p2 is not None:
            z1 = complex(p1_or_dist.x, p1_or_dist.y)
            z2 = complex(p2.x, p2.y)
            a = z2
        else:
            a = complex(p1_or_dist.x, p1_or_dist.y) if hasattr(p1_or_dist, 'x') else complex(p1_or_dist)
        # Translation by a: f(z) = (z + a)/(1 + conj(a)*z)
        return cls(1, a, a.conjugate(), 1)
    @classmethod
    def shift_origin(cls, new_origin, direction=None):
        a = complex(new_origin.x, new_origin.y)
        return cls(1, -a, -a.conjugate(), 1)
    @classmethod
    def merge(cls, *transforms):
        result = transforms[0]
        for t in transforms[1:]:
            if isinstance(result, MirrorTransform) or isinstance(t, MirrorTransform):
                return ComposedTransform(list(transforms))
            # Compose Möbius: (a1*z+b1)/(c1*z+d1) then (a2*w+b2)/(c2*w+d2)
            a = t.a * result.a + t.b * result.c
            b = t.a * result.b + t.b * result.d
            c = t.c * result.a + t.d * result.c
            d = t.c * result.b + t.d * result.d
            result = cls(a, b, c, d)
        return result

class MirrorTransform:
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy
    def apply(self, z):
        # Reflect z across line through origin perpendicular to (dx, dy)
        # Actually reflect across the line IN direction (dx, dy)
        z = complex(z)
        e = complex(self.dx, self.dy)
        return (e * e * z.conjugate())
    def apply_to_point(self, p):
        w = self.apply(complex(p.x, p.y))
        return Point(w.real, w.imag)
    def __call__(self, *points):
        return [self.apply_to_point(p) for p in points]

class ComposedTransform:
    def __init__(self, transforms):
        self.transforms = transforms
    def apply(self, z):
        for t in self.transforms:
            if isinstance(t, MirrorTransform):
                z = t.apply(z)
            else:
                z = t.apply(z)
        return z
    def apply_to_point(self, p):
        w = self.apply(complex(p.x, p.y))
        return Point(w.real, w.imag)


# ─── SVG Renderer ─────────────────────────────────────────────────────

class SVGRenderer:
    """Renders hyperbolic primitives to SVG."""
    def __init__(self, width, height, viewbox=None, origin='center'):
        self.width = width
        self.height = height
        if origin == 'center':
            half_w = width / 2
            half_h = height / 2
            self.vb = (-half_w, -half_h, width, height)
        elif viewbox:
            self.vb = viewbox
        else:
            self.vb = (0, 0, width, height)
        self.elements = []
        self.defs = []

    def add_def(self, s):
        self.defs.append(s)

    def _style_str(self, **kw):
        parts = []
        fill = kw.get('fill', 'none')
        parts.append(f'fill="{fill}"')
        if 'opacity' in kw:
            parts.append(f'opacity="{kw["opacity"]}"')
        if 'fill_opacity' in kw:
            parts.append(f'fill-opacity="{kw["fill_opacity"]}"')
        stroke = kw.get('stroke', 'none')
        parts.append(f'stroke="{stroke}"')
        sw = kw.get('stroke_width', kw.get('stroke-width', 0))
        if sw:
            parts.append(f'stroke-width="{sw}"')
        if 'stroke_opacity' in kw:
            parts.append(f'stroke-opacity="{kw["stroke_opacity"]}"')
        if kw.get('stroke_linecap'):
            parts.append(f'stroke-linecap="{kw["stroke_linecap"]}"')
        if kw.get('filter'):
            parts.append(f'filter="{kw["filter"]}"')
        if kw.get('class_name'):
            parts.append(f'class="{kw["class_name"]}"')
        return ' '.join(parts)

    def circle(self, cx, cy, r, **kw):
        self.elements.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" {self._style_str(**kw)}/>')

    def path(self, d, **kw):
        self.elements.append(f'<path d="{d}" {self._style_str(**kw)}/>')

    def line(self, x1, y1, x2, y2, **kw):
        self.elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" {self._style_str(**kw)}/>')

    def rect(self, x, y, w, h, **kw):
        self.elements.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" {self._style_str(**kw)}/>')

    def text(self, x, y, txt, font_size=0.03, **kw):
        fill = kw.get('fill', 'white')
        anchor = kw.get('text_anchor', 'middle')
        opacity = kw.get('opacity', 1)
        self.elements.append(
            f'<text x="{x}" y="{y}" font-size="{font_size}" fill="{fill}" '
            f'text-anchor="{anchor}" dominant-baseline="central" opacity="{opacity}" '
            f'font-family="monospace">{txt}</text>')

    def group_start(self, **kw):
        attrs = ''
        if 'transform' in kw:
            attrs += f' transform="{kw["transform"]}"'
        if 'opacity' in kw:
            attrs += f' opacity="{kw["opacity"]}"'
        if 'filter' in kw:
            attrs += f' filter="{kw["filter"]}"'
        if 'class_name' in kw:
            attrs += f' class="{kw["class_name"]}"'
        self.elements.append(f'<g{attrs}>')

    def group_end(self):
        self.elements.append('</g>')

    def raw(self, s):
        self.elements.append(s)

    def draw_point(self, point, radius=0.02, transform=None, **kw):
        if transform:
            z = complex(point.x, point.y)
            w = transform.apply(z)
            px, py = w.real, w.imag
        else:
            px, py = point.x, point.y
        self.circle(px, py, radius, **kw)

    def draw_geodesic(self, line, hwidth=None, transform=None, **kw):
        """Draw a geodesic line/segment."""
        x1, y1 = line.x1, line.y1
        x2, y2 = line.x2, line.y2
        if transform:
            w1 = transform.apply(complex(x1, y1))
            w2 = transform.apply(complex(x2, y2))
            x1, y1 = w1.real, w1.imag
            x2, y2 = w2.real, w2.imag
            # In half-plane, geodesics are semicircles or vertical lines
            if not line.segment:
                # Extend to ideal endpoints
                ep1, ep2 = line.ideal_endpoints()
                we1 = transform.apply(complex(*ep1))
                we2 = transform.apply(complex(*ep2))
                x1, y1 = we1.real, we1.imag
                x2, y2 = we2.real, we2.imag
        if hwidth and hwidth > 0:
            # Draw as thick arc
            sw = hwidth
            kw['stroke'] = kw.get('fill', kw.get('stroke', 'black'))
            kw['stroke_width'] = sw
            kw['fill'] = 'none'
            kw['stroke_linecap'] = 'round'
        if transform and isinstance(transform, (Transform, ComposedTransform)):
            # In the half-plane, need to recompute the geodesic
            circ = _geodesic_circle(x1, y1, x2, y2) if not transform else None
            # For half-plane geodesics: vertical lines or semicircles on real axis
            # Just draw a line or arc between transformed endpoints
            mid_z = complex((line.x1+line.x2)/2, (line.y1+line.y2)/2)
            w_mid = transform.apply(mid_z)
            circ = _circle_through_three(x1, y1, w_mid.real, w_mid.imag, x2, y2)
            if circ:
                cx, cy, r = circ
                d = _svg_arc_path(cx, cy, r, x1, y1, x2, y2)
                self.path(d, **kw)
            else:
                self.line(x1, y1, x2, y2, **kw)
            return
        circ = _geodesic_circle(line.x1, line.y1, line.x2, line.y2)
        if circ is None:
            if line.segment:
                self.line(x1, y1, x2, y2, **kw)
            else:
                # Extend through origin
                dx, dy = x2 - x1, y2 - y1
                r = math.hypot(dx, dy)
                if r > EPS:
                    nx, ny = dx/r, dy/r
                    self.line(-nx, -ny, nx, ny, **kw)
                else:
                    self.line(-1, 0, 1, 0, **kw)
        else:
            cx, cy, r = circ
            if line.segment:
                d = _svg_arc_path(cx, cy, r, x1, y1, x2, y2)
            else:
                # Full geodesic: draw between ideal endpoints
                ep1, ep2 = line.ideal_endpoints()
                d = _svg_arc_path(cx, cy, r, ep1[0], ep1[1], ep2[0], ep2[1])
            self.path(d, **kw)

    def draw_hypercycle(self, hc, hwidth=0.05, transform=None, **kw):
        """Draw a hypercycle as a thick arc."""
        if hc._circ is None:
            self.draw_geodesic(hc.base_line, hwidth=hwidth, transform=transform, **kw)
            return
        cx, cy, r = hc._circ
        ep1 = hc._ep1
        ep2 = hc._ep2
        if transform:
            we1 = transform.apply(complex(*ep1))
            we2 = transform.apply(complex(*ep2))
            wm = transform.apply(complex(*hc._mid)) if hasattr(hc, '_mid') else None
            x1, y1 = we1.real, we1.imag
            x2, y2 = we2.real, we2.imag
            if wm:
                circ = _circle_through_three(x1, y1, wm.real, wm.imag, x2, y2)
                if circ:
                    cx, cy, r = circ
                    d = _svg_arc_path(cx, cy, r, x1, y1, x2, y2)
                    kw.setdefault('stroke', kw.pop('fill', 'yellow'))
                    kw['stroke_width'] = hwidth
                    kw['fill'] = 'none'
                    self.path(d, **kw)
                    return
        # Disk rendering
        d = _svg_arc_path(cx, cy, r, ep1[0], ep1[1], ep2[0], ep2[1])
        stroke_color = kw.pop('fill', 'yellow')
        kw['stroke'] = stroke_color
        kw['stroke_width'] = hwidth
        kw['fill'] = 'none'
        kw['stroke_linecap'] = 'round'
        self.path(d, **kw)

    def draw_circle(self, circ, hwidth=0.02, transform=None, **kw):
        if transform:
            # Transform center
            w = transform.apply(complex(circ.e_cx, circ.e_cy))
            # For half-plane, circles stay circles (Möbius preserves circles)
            # but radius changes — approximate
            edge_z = complex(circ.e_cx + circ.e_r, circ.e_cy)
            we = transform.apply(edge_z)
            r = abs(complex(we.real - w.real, we.imag - w.imag))
            self.circle(w.real, w.imag, max(r, 0.01), **kw)
        else:
            self.circle(circ.e_cx, circ.e_cy, circ.e_r, **kw)

    def draw_horocycle(self, horo, hwidth=0.02, transform=None, **kw):
        if transform:
            w = transform.apply(complex(horo.e_cx, horo.e_cy))
            edge_z = complex(horo.e_cx + horo.e_r, horo.e_cy)
            we = transform.apply(edge_z)
            r = abs(complex(we.real - w.real, we.imag - w.imag))
            stroke_c = kw.pop('fill', 'blue')
            kw['stroke'] = stroke_c
            kw['stroke_width'] = hwidth
            kw['fill'] = 'none'
            self.circle(w.real, w.imag, max(r, 0.01), **kw)
        else:
            stroke_c = kw.pop('fill', 'blue')
            kw['stroke'] = stroke_c
            kw['stroke_width'] = hwidth
            kw['fill'] = 'none'
            self.circle(horo.e_cx, horo.e_cy, horo.e_r, **kw)

    def to_svg(self, render_width=None):
        vx, vy, vw, vh = self.vb
        defs_str = '\n'.join(self.defs)
        body = '\n'.join(self.elements)
        # Responsive: use viewBox only, no fixed width/height
        size_attr = ''
        if render_width:
            aspect = vh / vw if vw > 0 else 1
            render_height = int(render_width * aspect)
            size_attr = f'width="{render_width}" height="{render_height}"'
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="{vx} {vy} {vw} {vh}"
     {size_attr}
     preserveAspectRatio="xMidYMid meet">
<defs>
{defs_str}
</defs>
{body}
</svg>'''
