"""Spatial pattern diagnostics: nearest-neighbour G, empty-space F, J = (1-G)/(1-F),
and the Clark-Evans aggregation index.

G and F are plain empirical CDFs without edge correction; in a large window
the bias is small, in a small one both are pulled towards longer distances.
Under complete spatial randomness (CSR) with intensity lambda,
G(r) = F(r) = 1 - exp(-lambda * pi * r^2) and J(r) = 1.
J < 1 indicates clustering, J > 1 regularity.
"""

import numpy as np
from scipy.spatial import cKDTree

from .geometry import points_inside

F_TEST_POINTS = 10_000


def nearest_neighbour_distances(x, y):
    """Distance from every point to its nearest other point."""
    if len(x) < 2:
        return np.full(len(x), np.nan)
    dist, _ = cKDTree(np.column_stack([x, y])).query(np.column_stack([x, y]), k=2)
    return dist[:, 1]


def clark_evans(x, y, area, perimeter):
    """Clark-Evans index R with Donnelly's edge correction.

    R = mean nearest-neighbour distance / its expectation under CSR,
    E = 0.5 sqrt(A/n) + (0.0514 + 0.041/sqrt(n)) P/n.
    R < 1 clustered, R near 1 random, R > 1 regular.  NaN for n < 2.
    """
    n = len(x)
    if n < 2 or area <= 0:
        return float("nan")
    expected = 0.5 * np.sqrt(area / n) + (0.0514 + 0.041 / np.sqrt(n)) * perimeter / n
    return float(np.mean(nearest_neighbour_distances(x, y)) / expected)


def r_values(n, area, count=60):
    """Distances 0 .. 1/sqrt(lambda) at which to evaluate G, F and J."""
    lam = n / area if area > 0 else 0.0
    r_max = 1.0 / np.sqrt(lam) if lam > 0 else 1.0
    return np.linspace(0.0, r_max, count)


def g_function(x, y, r):
    """Empirical nearest-neighbour distance distribution G(r) (no edge correction)."""
    d = np.sort(nearest_neighbour_distances(x, y))
    if len(d) == 0 or np.isnan(d).all():
        return np.full(len(r), np.nan)
    return np.searchsorted(d, r, side="right") / len(d)


def grid_points(region, count=F_TEST_POINTS):
    """A regular grid of about ``count`` points inside ``region``."""
    minx, miny, maxx, maxy = region.bounds
    if region.is_empty or region.area <= 0:
        return np.empty(0), np.empty(0)
    step = np.sqrt(region.area / count)
    gx, gy = np.meshgrid(np.arange(minx + step / 2, maxx, step),
                         np.arange(miny + step / 2, maxy, step))
    gx, gy = gx.ravel(), gy.ravel()
    keep = points_inside(region, gx, gy)
    return gx[keep], gy[keep]


def f_function(x, y, region, r):
    """Empirical empty-space function F(r) from a grid of test points (no edge correction)."""
    tx, ty = grid_points(region)
    if len(x) == 0 or len(tx) == 0:
        return np.full(len(r), np.nan)
    d, _ = cKDTree(np.column_stack([x, y])).query(np.column_stack([tx, ty]))
    return np.searchsorted(np.sort(d), r, side="right") / len(d)


def pattern_functions(x, y, region, r=None):
    """G, F and J of the points in ``region`` with the CSR reference curves."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    area = region.area
    if r is None:
        r = r_values(len(x), area)
    g = g_function(x, y, r)
    f = f_function(x, y, region, r)
    with np.errstate(divide="ignore", invalid="ignore"):
        j = np.where(f < 1.0, (1.0 - g) / (1.0 - f), np.nan)
    lam = len(x) / area if area > 0 else 0.0
    csr = 1.0 - np.exp(-lam * np.pi * r ** 2)
    return {"r": r, "G": g, "F": f, "J": j, "csr": csr, "n": len(x), "area": area}
