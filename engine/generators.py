"""Automatic generators: one click adds ordinary, editable layers.

Each generator takes a recipe and a ``numpy.random.Generator`` and returns a
list of new layer dicts (ids not yet used in the recipe).  Nothing here is
special at generation time: the layers are plain recipe layers.
"""

import copy

import numpy as np
from scipy.spatial import Voronoi
import shapely
from shapely.geometry import Point, Polygon

from .geometry import window_geometry
from .presets import PRESETS
from .processes import uniform_points
from .recipe import next_layer_id, validate_recipe

GENERATORS = ("mosaic", "gaps", "trails", "understorey", "beetle_patches")
MOSAIC_WEIGHTS = {
    "young_spruce": 0.25,
    "pine_heath": 0.20,
    "mixed_birch_spruce": 0.30,
    "old_growth": 0.15,
    "seedling_stand": 0.05,
    "managed_rows": 0.05,
}


def _coords(geom_coords):
    pts = [[round(float(x), 2), round(float(y), 2)] for x, y in geom_coords]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def _with_ids(recipe, layers):
    """Give new layers ids that do not clash with the recipe's layers."""
    existing = list(recipe.get("layers", []))
    for layer in layers:
        layer["id"] = next_layer_id({"layers": existing})
        existing.append(layer)
    return layers


def mosaic(recipe, rng, n_patches=8, weights=None):
    """Voronoi patches over the window, each a replace-mode layer with a random preset."""
    window = window_geometry(recipe["window"])
    weights = weights or MOSAIC_WEIGHTS
    names = [k for k, w in weights.items() if w > 0]
    probs = np.array([weights[k] for k in names], dtype=float)
    x, y = uniform_points(window, max(int(n_patches), 2), rng)
    minx, miny, maxx, maxy = window.bounds
    cx, cy, size = (minx + maxx) / 2, (miny + maxy) / 2, max(maxx - minx, maxy - miny)
    far = np.array([[cx - 10 * size, cy], [cx + 10 * size, cy], [cx, cy - 10 * size], [cx, cy + 10 * size]])
    vor = Voronoi(np.vstack([np.column_stack([x, y]), far]))
    presets = rng.choice(names, size=len(x), p=probs / probs.sum())
    layers = []
    for i, preset in enumerate(presets):
        region = vor.regions[vor.point_region[i]]
        if -1 in region or len(region) < 3:
            continue
        with np.errstate(invalid="ignore"):
            cell = Polygon(vor.vertices[region]).intersection(window)
        for part in shapely.get_parts(cell):
            if isinstance(part, Polygon) and part.area > 1.0:
                layers.append({
                    "name": f"Patch {len(layers) + 1}: {preset}",
                    "type": PRESETS[preset]["type"], "mode": "replace", "preset": str(preset),
                    "shape": {"kind": "polygon", "vertices": _coords(part.exterior.coords)},
                })
    return _with_ids(recipe, layers)


def _random_centre(window, margin, rng, avoid=(), tries=200):
    inner = window.buffer(-margin)
    region = inner if inner.area > 0 else window
    for _ in range(tries):
        x, y = uniform_points(region, 1, rng)
        centre = Point(x[0], y[0])
        if all(centre.distance(c) > r + margin for c, r in avoid):
            return centre
    return None


def gaps(recipe, rng, n_gaps=5, radius_min=10.0, radius_max=25.0, edge_density=1500.0):
    """Circular clearings with a ring of small regenerating trees around each edge."""
    window = window_geometry(recipe["window"])
    placed, layers = [], []
    for k in range(int(n_gaps)):
        radius = float(rng.uniform(radius_min, radius_max))
        centre = _random_centre(window, radius, rng, avoid=placed)
        if centre is None:
            continue
        placed.append((centre, radius))
        c = [round(centre.x, 2), round(centre.y, 2)]
        ring = centre.buffer(radius, quad_segs=12).exterior.coords
        points = _coords(ring)
        points.append(points[0])
        layers.append({"name": f"Gap {k + 1}", "type": "clear",
                       "shape": {"kind": "circle", "center": c, "radius": round(radius, 2)}})
        layers.append({"name": f"Gap {k + 1} edge regeneration", "type": "random", "mode": "add",
                       "preset": "seedling_stand", "params": {"density": float(edge_density)},
                       "shape": {"kind": "corridor", "points": points,
                                 "width": round(max(4.0, 0.4 * radius), 2)}})
    return _with_ids(recipe, layers)


def _boundary_point(window, rng):
    ring = window.exterior if isinstance(window, Polygon) else max(
        shapely.get_parts(window), key=lambda g: g.area).exterior
    return ring.interpolate(float(rng.uniform(0, ring.length)))


def trails(recipe, rng, n_trails=1, width=4.0, wiggle=0.05):
    """Corridors crossing the window from edge to edge, with gentle bends."""
    window = window_geometry(recipe["window"])
    minx, miny, maxx, maxy = window.bounds
    span = max(maxx - minx, maxy - miny)
    layers = []
    for k in range(int(n_trails)):
        a = _boundary_point(window, rng)
        b = _boundary_point(window, rng)
        for _ in range(100):
            if a.distance(b) > 0.8 * span:
                break
            b = _boundary_point(window, rng)
        start, end = np.array(a.coords[0]), np.array(b.coords[0])
        direction = (end - start) / max(np.linalg.norm(end - start), 1e-9)
        normal = np.array([-direction[1], direction[0]])
        length = np.linalg.norm(end - start)
        t = np.linspace(0.0, 1.0, 7)
        offsets = rng.normal(0.0, wiggle * length, len(t)) * np.sin(np.pi * t)
        pts = start + np.outer(t, end - start) + np.outer(offsets, normal)
        pts[0] -= direction * width
        pts[-1] += direction * width
        layers.append({"name": f"Trail {k + 1}", "type": "clear",
                       "shape": {"kind": "corridor", "points": _coords(pts), "width": float(width)}})
    return _with_ids(recipe, layers)


def understorey(recipe, rng=None, density=250.0, shape=None):
    """Small, mostly spruce trees added beneath the existing canopy (add mode)."""
    per_cluster = 10.0
    layer = {
        "name": "Understorey", "type": "clustered", "mode": "add",
        "params": {"cluster_density": round(density / per_cluster, 3),
                   "trees_per_cluster": per_cluster, "cluster_radius": 4.0},
        "composition": {"species": {"pine": 0.05, "spruce": 0.75, "broadleaved": 0.20},
                        "size_class": "young", "dead_share": 0.01},
        "shape": copy.deepcopy(shape) if shape else {"kind": "window"},
    }
    return _with_ids(recipe, [layer])


def beetle_patches(recipe, rng, n_foci=3, patches_per_focus=4, spread=40.0,
                   radius_min=8.0, radius_max=20.0, dead_share=0.7):
    """Clustered groups of small replace-mode patches of mature, mostly dead spruce."""
    window = window_geometry(recipe["window"])
    layers = []
    for f in range(int(n_foci)):
        focus = _random_centre(window, spread, rng)
        if focus is None:
            continue
        for _ in range(int(patches_per_focus)):
            cx, cy = focus.x + rng.normal(0, spread), focus.y + rng.normal(0, spread)
            radius = float(rng.uniform(radius_min, radius_max))
            if not window.contains(Point(cx, cy)):
                continue
            layers.append({
                "name": f"Beetle patch {len(layers) + 1} (focus {f + 1})",
                "type": "clustered", "mode": "replace",
                "params": {"cluster_density": 40.0, "trees_per_cluster": 12.0, "cluster_radius": 4.0},
                "composition": {"species": {"pine": 0.05, "spruce": 0.90, "broadleaved": 0.05},
                                "size_class": "mature", "dead_share": float(dead_share)},
                "shape": {"kind": "circle", "center": [round(cx, 2), round(cy, 2)],
                          "radius": round(radius, 2)},
            })
    return _with_ids(recipe, layers)


def apply_generator(recipe, name, rng, **options):
    """Return a validated copy of ``recipe`` with the generator's layers appended."""
    funcs = {"mosaic": mosaic, "gaps": gaps, "trails": trails,
             "understorey": understorey, "beetle_patches": beetle_patches}
    if name not in funcs:
        raise ValueError(f"unknown generator {name!r}; choose one of {', '.join(GENERATORS)}")
    recipe = validate_recipe(recipe)
    recipe["layers"].extend(funcs[name](recipe, rng, **options))
    return validate_recipe(recipe)
