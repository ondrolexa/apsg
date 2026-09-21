"""Small helpers shared by ``_axes.py`` and ``_stereogrid.py``.

Kept in their own module (rather than living in ``_axes.py``) so
``_stereogrid.py`` can use them without an import cycle (``_axes.py``
imports ``StereoGrid`` from ``_stereogrid.py``).
"""

import numpy as np


def _as_vectors(xyz):
    """Normalize array-like input (shape (3,) or (N, 3)) to a unit-vector
    (N, 3) array. Anything numpy can coerce with ``asarray`` works, which
    includes apsg's ``Vector3``/``Vector3Set`` (they implement
    ``__array__``) without this module depending on apsg."""
    v = np.atleast_2d(np.asarray(xyz, dtype=float))
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / norm


def _crosses_geodesic(p, q, a, b):
    """Boolean array telling which of the great-circle arcs ``a[i] -> b[i]``
    cross the (minor) great-circle arc ``p -> q``. All vectors are unit."""
    n1 = np.cross(p, q)
    n2 = np.cross(a, b)
    d = np.cross(n1, n2)  # direction of the two intersections of the great circles
    norm = np.linalg.norm(d, axis=1)
    ok = norm > 1e-12
    d = d / np.where(ok, norm, 1.0)[:, None]
    hit = np.zeros(len(a), dtype=bool)
    for sign in (1.0, -1.0):
        c = sign * d
        on_pq = (np.cross(p, c) @ n1 > 0) & (np.cross(c, q) @ n1 > 0)
        on_ab = (np.einsum("ij,ij->i", np.cross(a, c), n2) > 0) & (
            np.einsum("ij,ij->i", np.cross(c, b), n2) > 0
        )
        hit |= ok & on_pq & on_ab
    return hit


def _ring_contains(ring, point):
    """Whether ``point`` lies on the side of the closed great-circle ring that
    does not contain the zenith ``(0, 0, -1)``: the great-circle arc from
    ``point`` to the zenith crosses the ring an odd number of times."""
    zenith = np.array([0.0, 0.0, -1.0])
    point = _as_vectors(point)[0]
    ring_end = np.roll(ring, -1, axis=0)
    if point[2] > 0.9998:
        # near the nadir the arc to the zenith is almost 180 degrees and ill
        # defined: go there via a point on the horizon instead
        east = np.array([1.0, 0.0, 0.0])
        crossings = _crosses_geodesic(point, east, ring, ring_end).sum()
        crossings += _crosses_geodesic(east, zenith, ring, ring_end).sum()
    else:
        crossings = _crosses_geodesic(point, zenith, ring, ring_end).sum()
    return int(crossings) % 2 == 1


def _clip_ring_to_hemisphere(ring, step=1.0):
    """Clip the region bounded by a closed ring of unit vectors to the visible
    hemisphere (``z >= 0``), for filling it in the projection.

    The region is the side of the ring not containing the zenith (the
    projection's singular point). Where the ring runs through the hidden
    hemisphere, its excursion is replaced by the stretch of the primitive
    circle (``z = 0``) that belongs to the region.

    Args:
        ring: (N, 3) vectors, densely sampled boundary, closed implicitly.
        step (float): sampling of the primitive circle in degrees.

    Returns:
        list of (M, 3) arrays: closed boundary loops of the visible region
        (empty if nothing of the region is visible).
    """
    v = _as_vectors(ring).copy()
    v[np.abs(v[:, 2]) < 1e-9, 2] = 0.0  # points on the horizon count as visible
    v = _as_vectors(v)
    visible = v[:, 2] >= 0
    if visible.all():
        return [v]
    if not visible.any():
        # the region is either the whole visible hemisphere or none of it
        rim = np.radians(np.arange(0.0, 360.0, step))
        disk = np.column_stack([np.cos(rim), np.sin(rim), np.zeros_like(rim)])
        return [disk] if _ring_contains(v, [0.0, 0.0, 1.0]) else []

    n = len(v)
    nxt = np.roll(np.arange(n), -1)
    edges = np.flatnonzero(visible != visible[nxt])
    # rotate so that the crossings start with an entry (hidden -> visible)
    if visible[edges[0]]:
        edges = np.roll(edges, -1)
    nodes = []
    for i in edges:  # the point where edge i -> i+1 meets the horizon
        a, b = v[i], v[nxt[i]]
        t = a[2] / (a[2] - b[2])
        c = a + t * (b - a)
        nodes.append(c / np.linalg.norm(c))
    nodes = np.array(nodes)
    m = len(edges) // 2
    # visible runs: node 2r (entry) -> vertices -> node 2r + 1 (exit)
    runs = []
    for r in range(m):
        i0, i1 = edges[2 * r], edges[2 * r + 1]
        idx = np.arange(i0 + 1, i1 + 1 + (n if i1 < i0 else 0)) % n
        runs.append(np.vstack([nodes[2 * r], v[idx], nodes[2 * r + 1]]))

    # arcs of the primitive circle between neighbouring crossings alternate
    # inside/outside the region; the inside ones connect the runs
    phi = np.arctan2(nodes[:, 1], nodes[:, 0])
    order = np.argsort(phi)
    partner, sweep = {}, {}  # neighbouring node, signed azimuth span of the arc
    for k in range(len(order)):
        a, b = order[k], order[(k + 1) % len(order)]
        width = (phi[b] - phi[a]) % (2 * np.pi)
        mid = phi[a] + width / 2
        if _ring_contains(v, [np.cos(mid), np.sin(mid), 1e-6]):
            partner[a], partner[b] = b, a
            sweep[a, b], sweep[b, a] = width, -width

    def rim_arc(a, b):
        """Points on the primitive circle from node a to node b along the
        inside arc between them."""
        count = max(2, int(np.ceil(abs(sweep[a, b]) / np.radians(step))) + 1)
        ang = phi[a] + np.linspace(0.0, sweep[a, b], count)
        return np.column_stack([np.cos(ang), np.sin(ang), np.zeros(count)])

    loops, seen = [], set()
    for start in range(len(nodes)):
        if start in seen or start not in partner:
            continue
        loop, cur = [], start
        while True:
            seen.add(cur)
            run = runs[cur // 2] if cur % 2 == 0 else runs[cur // 2][::-1]
            end = cur + 1 if cur % 2 == 0 else cur - 1
            seen.add(end)
            loop.append(run)
            nxt_node = partner.get(end)
            if nxt_node is None:
                break
            loop.append(rim_arc(end, nxt_node))
            cur = nxt_node
            if cur == start:
                break
        loops.append(np.vstack(loop))
    return loops
