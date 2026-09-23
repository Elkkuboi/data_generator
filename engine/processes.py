"""Point processes that place trees inside a shape.

Densities are in trees per hectare, distances in metres, angles in degrees
counter-clockwise from the +x axis (east).  Every function takes the shape
as a prepared shapely geometry and an explicit ``numpy.random.Generator``,
and returns coordinate arrays ``x, y`` of points inside the shape.
"""

import numpy as np
from scipy.spatial import cKDTree

from .geometry import points_inside

HA = 10_000.0
# Random sequential adsorption of discs jams at an area coverage of ~0.547;
# hexagonal packing is the absolute limit for a minimum distance.
RSA_JAMMING_COVERAGE = 0.547


def _box_poisson(bounds, intensity_m2, rng):
    minx, miny, maxx, maxy = bounds
    n = rng.poisson(max(intensity_m2, 0.0) * (maxx - minx) * (maxy - miny))
    return rng.uniform(minx, maxx, n), rng.uniform(miny, maxy, n)


def random_points(geom, density, rng):
    """Homogeneous Poisson process with ``density`` trees/ha inside ``geom``."""
    if geom.is_empty or density <= 0:
        return np.empty(0), np.empty(0)
    x, y = _box_poisson(geom.bounds, density / HA, rng)
    keep = points_inside(geom, x, y)
    return x[keep], y[keep]


def uniform_points(geom, n, rng):
    """Exactly ``n`` independent uniform points inside ``geom`` (rejection sampling)."""
    if geom.is_empty or n <= 0:
        return np.empty(0), np.empty(0)
    minx, miny, maxx, maxy = geom.bounds
    acceptance = geom.area / ((maxx - minx) * (maxy - miny))
    xs, ys, have = [], [], 0
    for _ in range(1000):
        m = int((n - have) / acceptance * 1.1) + 16
        x = rng.uniform(minx, maxx, m)
        y = rng.uniform(miny, maxy, m)
        keep = points_inside(geom, x, y)
        xs.append(x[keep])
        ys.append(y[keep])
        have += int(keep.sum())
        if have >= n:
            break
    return np.concatenate(xs)[:n], np.concatenate(ys)[:n]


def clustered_points(geom, cluster_density, trees_per_cluster, cluster_radius, rng):
    """Thomas cluster process.

    Parents: Poisson with ``cluster_density`` per ha.  Each parent gets a
    Poisson(``trees_per_cluster``) number of trees displaced by an isotropic
    Gaussian with standard deviation ``cluster_radius`` (the Thomas sigma,
    spatstat's ``scale``).  Parents are drawn in the shape's bounding box
    widened by 4 sigma, so clusters are not cut at the shape's edge.
    """
    if geom.is_empty or cluster_density <= 0 or trees_per_cluster <= 0:
        return np.empty(0), np.empty(0)
    pad = 4.0 * cluster_radius
    minx, miny, maxx, maxy = geom.bounds
    px, py = _box_poisson((minx - pad, miny - pad, maxx + pad, maxy + pad),
                          cluster_density / HA, rng)
    counts = rng.poisson(trees_per_cluster, len(px))
    x = np.repeat(px, counts) + rng.normal(0.0, cluster_radius, counts.sum())
    y = np.repeat(py, counts) + rng.normal(0.0, cluster_radius, counts.sum())
    keep = points_inside(geom, x, y)
    return x[keep], y[keep]


def gradient_points(geom, density_start, density_end, direction, rng):
    """Inhomogeneous Poisson process whose density changes linearly.

    The density is ``density_start`` at the back of the shape and
    ``density_end`` at its front, measured along ``direction``.
    """
    top = max(density_start, density_end)
    if geom.is_empty or top <= 0:
        return np.empty(0), np.empty(0)
    x, y = random_points(geom, top, rng)
    theta = np.radians(direction)
    ux, uy = np.cos(theta), np.sin(theta)
    minx, miny, maxx, maxy = geom.bounds
    corners = np.array([[minx, miny], [maxx, miny], [minx, maxy], [maxx, maxy]])
    proj = corners @ np.array([ux, uy])
    t = (x * ux + y * uy - proj.min()) / max(proj.max() - proj.min(), 1e-9)
    density = density_start + (density_end - density_start) * np.clip(t, 0.0, 1.0)
    keep = rng.random(len(x)) < density / top
    return x[keep], y[keep]


def rows_points(geom, row_spacing, tree_spacing, angle, jitter, rng, keep_fraction=1.0):
    """Planted rows: a rotated grid with a random phase and Gaussian jitter.

    Rows run along ``angle``; ``row_spacing`` separates rows and
    ``tree_spacing`` separates trees within a row.  ``jitter`` is the
    standard deviation (m) of the planting error.  ``keep_fraction`` < 1
    keeps a random subset of planting spots (used by draft mode).
    """
    if geom.is_empty:
        return np.empty(0), np.empty(0)
    theta = np.radians(angle)
    along = np.array([np.cos(theta), np.sin(theta)])
    across = np.array([-np.sin(theta), np.cos(theta)])
    minx, miny, maxx, maxy = geom.bounds
    pad = 4.0 * jitter + max(row_spacing, tree_spacing)
    corners = np.array([[minx - pad, miny - pad], [maxx + pad, miny - pad],
                        [minx - pad, maxy + pad], [maxx + pad, maxy + pad]])
    a_proj, c_proj = corners @ along, corners @ across
    a = a_proj.min() + rng.uniform(0, tree_spacing) + np.arange(
        0.0, a_proj.max() - a_proj.min(), tree_spacing)
    c = c_proj.min() + rng.uniform(0, row_spacing) + np.arange(
        0.0, c_proj.max() - c_proj.min(), row_spacing)
    aa, cc = np.meshgrid(a, c)
    aa, cc = aa.ravel(), cc.ravel()
    x = aa * along[0] + cc * across[0] + rng.normal(0.0, jitter, aa.size)
    y = aa * along[1] + cc * across[1] + rng.normal(0.0, jitter, aa.size)
    keep = points_inside(geom, x, y)
    if keep_fraction < 1.0:
        keep &= rng.random(aa.size) < keep_fraction
    return x[keep], y[keep]


def max_regular_density(min_distance):
    """Hexagonal-packing limit (trees/ha) for a minimum distance: never reachable."""
    return 2.0 / (np.sqrt(3.0) * min_distance ** 2) * HA


def practical_regular_density(min_distance):
    """Density (trees/ha) where random sequential inhibition jams."""
    return RSA_JAMMING_COVERAGE / (np.pi * min_distance ** 2 / 4.0) * HA


def greedy_inhibition(x, y, min_distance, n_max=None):
    """Indices of points kept by sequential inhibition in array order.

    Point i is kept if no earlier kept point lies within ``min_distance``
    (distances equal to it count as violations).  This is the result of
    placing the points one by one, computed in vectorised rounds: in each
    round every undecided point with no undecided earlier neighbour is
    accepted and its neighbours rejected.  Only the first ``n_max`` kept
    points are returned.
    """
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=np.intp)
    if min_distance <= 0:
        kept = np.arange(n)
        return kept if n_max is None else kept[:n_max]
    pairs = cKDTree(np.column_stack([x, y])).query_pairs(min_distance, output_type="ndarray")
    first, second = np.sort(pairs, axis=1).T if len(pairs) else (np.empty(0, int),) * 2
    state = np.zeros(n, dtype=np.int8)          # 0 undecided, 1 kept, -1 rejected
    while True:
        undecided = state == 0
        if not undecided.any():
            break
        live = undecided[first] & undecided[second]
        blocked = np.zeros(n, dtype=bool)
        blocked[second[live]] = True
        accept = undecided & ~blocked
        state[accept] = 1
        hit = accept[first] & (state[second] == 0)
        state[second[hit]] = -1
    kept = np.flatnonzero(state == 1)
    return kept if n_max is None else kept[:n_max]
