"""Sarkar-style tree embedding in the Poincaré disk.

Each spanning-tree edge has hyperbolic length `tau`. Children sit in a cone
facing away from the parent, with angle proportional to subtree size. The
root's children use the full circle. This is a hyperbolic embedding: edge
length is constant under the disk metric, unlike a polar plot of PageRank.
"""

from __future__ import annotations

import math


def mobius_add(z: complex, w: complex) -> complex:
    denom = 1 + z.conjugate() * w
    if abs(denom) < 1e-15:
        return 0j
    return (z + w) / denom


def hyperbolic_distance(z: complex, w: complex) -> float:
    denom = (1 - abs(z) ** 2) * (1 - abs(w) ** 2)
    if denom <= 1e-15:
        return float("inf")
    value = 1 + 2 * abs(z - w) ** 2 / denom
    return math.acosh(max(1.0, value))


def _toward_parent(parent: complex, local: complex) -> float:
    """Angle, in the child's tangent space, pointing back at the parent."""
    child = mobius_add(parent, local)
    if abs(child) < 1e-12:
        return math.atan2(-local.imag, -local.real)
    diff = parent - child
    denom = 1 - child.conjugate() * parent
    if abs(denom) < 1e-15:
        return math.atan2(-local.imag, -local.real)
    mapped = diff / denom
    return math.atan2(mapped.imag, mapped.real)


def subtree_sizes(root: str, children: dict[str, list[str]]) -> dict[str, int]:
    sizes: dict[str, int] = {}

    def measure(node: str) -> int:
        total = 1
        for child in children.get(node, ()):
            total += measure(child)
        sizes[node] = total
        return total

    measure(root)
    return sizes


def sarkar_embed(
    root: str,
    children: dict[str, list[str]],
    *,
    tau: float = 1.0,
) -> dict[str, complex]:
    """Place `root` at the origin and embed the tree with edge length `tau`."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    sizes = subtree_sizes(root, children)
    rho = math.tanh(tau / 2)
    positions: dict[str, complex] = {root: 0j}

    def place(node: str, incoming: float | None) -> None:
        kids = children.get(node, ())
        if not kids:
            return
        if incoming is None:
            cone_left = -math.pi
            cone_width = 2 * math.pi
        else:
            # Face away from the parent. A half-turn keeps sibling cones apart
            # at the edge lengths used for depth-2 and depth-3 curricula.
            cone_width = math.pi
            cone_left = incoming + math.pi - cone_width / 2
        weight = sum(sizes[child] for child in kids)
        cursor = cone_left
        for child in kids:
            span = cone_width * (sizes[child] / weight)
            mid = cursor + span / 2
            local = complex(rho * math.cos(mid), rho * math.sin(mid))
            positions[child] = mobius_add(positions[node], local)
            place(child, _toward_parent(positions[node], local))
            cursor += span

    place(root, None)
    return positions


def disk_to_half_plane(z: complex) -> complex:
    """Notebook map w = conj(i (1 + conj(z)) / (1 - conj(z))) onto the lower half-plane."""
    conj = z.conjugate()
    numer = 1 + conj
    denom = 1 - conj
    if abs(denom) < 1e-12:
        raise ValueError("point is on the boundary")
    quot = numer / denom
    rotated = 1j * quot
    return rotated.conjugate()


def children_from_parents(root: str, parent: dict[str, str | None]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {node: [] for node in parent}
    for node, par in parent.items():
        if node == root or par is None:
            continue
        children.setdefault(par, []).append(node)
    for node in children:
        children[node].sort()
    return children
