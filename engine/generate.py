"""Recipe + seed -> forest (a pandas DataFrame of trees plus ground truth).

Layers are applied in order.  An adding layer draws its points, assigns
attributes and places the trees (laser lattice for ITD trees, 0.01 m
elsewhere) inside its shape; in ``replace`` mode it first removes earlier
trees in the shape.  ``clear`` and ``thin`` act on the trees of all earlier
layers.  Each layer draws from its own random stream derived from the seed
and the layer id, so editing one layer does not reshuffle the others.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .attributes import assign_attributes, stem_volume
from .geometry import area_ha, points_touching, shape_geometry, snap_inside, window_geometry
from .presets import SPECIES_CODES
from .processes import (
    clustered_points,
    gradient_points,
    greedy_inhibition,
    random_points,
    rows_points,
    uniform_points,
)
from .recipe import ADDING_TYPES, RecipeError, layer_stream_key, resolve_layer, validate_recipe

COLUMNS = ("X", "Y", "H", "DBH", "Species", "Latvus_h", "Latvus_d", "Status", "Type", "V")
TRUTH_COLUMNS = ("layer_id", "layer_name", "layer_type", "preset")
LATTICE_STEP = 0.5           # m, grid of ITD coordinates
COORD_STEP = 0.01            # m, output precision
BACKGROUND_ID = "background"
_FIELDS = ("x", "y", "species", "dbh", "height", "crown_d", "crown_l", "dead", "itd", "layer")


@dataclass
class Forest:
    """A generated forest and everything needed to explain it."""

    trees: pd.DataFrame
    truth: pd.DataFrame
    recipe: dict
    window: object
    warnings: list = field(default_factory=list)
    layer_counts: dict = field(default_factory=dict)
    elapsed: float = 0.0
    density_scale: float = 1.0


# --------------------------------------------------------------------------
# Tree sets as dicts of parallel arrays
# --------------------------------------------------------------------------

def _empty():
    return {"x": np.empty(0), "y": np.empty(0), "species": np.empty(0, np.int8),
            "dbh": np.empty(0), "height": np.empty(0), "crown_d": np.empty(0),
            "crown_l": np.empty(0), "dead": np.empty(0, bool), "itd": np.empty(0, bool),
            "layer": np.empty(0, np.int32)}


def _take(trees, index):
    return {k: v[index] for k, v in trees.items()}


def _concat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in _FIELDS}


def _count(trees):
    return len(trees["x"])


# --------------------------------------------------------------------------
# Placing trees
# --------------------------------------------------------------------------

def _stream(seed, *key):
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def lattice_offset(seed):
    """The fixed random offset (multiple of 0.01 m) of this area's ITD lattice."""
    rng = _stream(seed, 0)
    return tuple(float(v) for v in rng.integers(0, int(LATTICE_STEP / COORD_STEP), 2) * COORD_STEP)


def _place(x, y, itd, geom, laser, lattice):
    """Final coordinates: ITD trees on the laser lattice, others on 0.01 m."""
    out_x, out_y, ok = np.empty(len(x)), np.empty(len(x)), np.zeros(len(x), bool)
    groups = [(itd, LATTICE_STEP, lattice), (~itd, COORD_STEP, (0.0, 0.0))] if laser \
        else [(np.ones(len(x), bool), COORD_STEP, (0.0, 0.0))]
    for mask, step, offset in groups:
        sx, sy, sok = snap_inside(x[mask], y[mask], step, offset, geom)
        out_x[mask], out_y[mask], ok[mask] = sx, sy, sok
    return out_x, out_y, ok


def _make_trees(x, y, composition, geom, rng, laser, lattice):
    attrs = assign_attributes(len(x), composition, rng, laser)
    px, py, ok = _place(x, y, attrs["itd"], geom, laser, lattice)
    trees = dict(attrs, x=px, y=py, layer=np.zeros(len(x), np.int32))
    return _take(trees, ok)


def _regular_trees(resolved, geom, rng, laser, lattice, scale):
    """Sequential inhibition on final coordinates, in batches, until the target count.

    Candidates are uniform points in random order; each is accepted if no
    earlier accepted tree lies within the minimum distance.  Distances are
    checked on the final (rounded or lattice-snapped) coordinates, so the
    output never violates the minimum distance.
    """
    p = resolved.params
    target = int(round(p["density"] * scale * area_ha(geom)))
    r = p["min_distance"]
    kept = []
    n_kept, drawn, acceptance = 0, 0, 1.0
    budget = 20 * target + 10_000
    while n_kept < target and drawn < budget:
        batch = min(int(1.3 * (target - n_kept) / acceptance) + 64, 400_000)
        x, y = uniform_points(geom, batch, rng)
        cand = _make_trees(x, y, resolved.composition, geom, rng, laser, lattice)
        drawn += batch
        if n_kept and r > 0 and _count(cand):
            done = _concat_list(kept)
            dist, _ = cKDTree(np.column_stack([done["x"], done["y"]])).query(
                np.column_stack([cand["x"], cand["y"]]), distance_upper_bound=r * 1.000001 + 1e-9)
            cand = _take(cand, dist > r)
        take = greedy_inhibition(cand["x"], cand["y"], r, target - n_kept)
        kept.append(_take(cand, take))
        n_kept += len(take)
        acceptance = max(len(take) / batch, 0.02)
    trees = _concat_list(kept) if kept else _empty()
    note = None
    if target and n_kept < 0.98 * target:
        note = (f"reached {n_kept / area_ha(geom):.0f} of the requested "
                f"{p['density'] * scale:g} trees/ha with min_distance {r:g} m")
    return trees, note


def _concat_list(parts):
    out = parts[0]
    for part in parts[1:]:
        out = _concat(out, part)
    return out


def _adding_trees(resolved, geom, rng, laser, lattice, scale):
    p, kind = resolved.params, resolved.type
    if kind == "regular":
        return _regular_trees(resolved, geom, rng, laser, lattice, scale)
    if kind == "random":
        x, y = random_points(geom, p["density"] * scale, rng)
    elif kind == "clustered":
        x, y = clustered_points(geom, p["cluster_density"] * scale, p["trees_per_cluster"],
                                p["cluster_radius"], rng)
    elif kind == "gradient":
        x, y = gradient_points(geom, p["density_start"] * scale, p["density_end"] * scale,
                               p["direction"], rng)
    elif kind == "rows":
        x, y = rows_points(geom, p["row_spacing"], p["tree_spacing"], p["angle"], p["jitter"],
                           rng, keep_fraction=scale)
    else:
        raise ValueError(f"not an adding layer type: {kind}")
    return _make_trees(x, y, resolved.composition, geom, rng, laser, lattice), None


def _thin_mask(trees, geom, params, rng):
    """Trees removed by a thin layer: a random ``fraction`` of the eligible ones.

    Random numbers are drawn only for eligible trees, so editing layers
    elsewhere does not change which trees a thin layer removes.
    """
    selected = points_touching(geom, trees["x"], trees["y"])
    if params.get("dbh_below") is not None:
        selected &= trees["dbh"] < params["dbh_below"]
    if params.get("dbh_above") is not None:
        selected &= trees["dbh"] > params["dbh_above"]
    index = np.flatnonzero(selected)
    remove = np.zeros(_count(trees), dtype=bool)
    remove[index[rng.random(len(index)) < params["fraction"]]] = True
    return remove


def _drop_duplicate_positions(trees):
    """Keep the first tree at each coordinate (laser detection merges such trees)."""
    if _count(trees) == 0:
        return trees
    key = np.round(trees["x"] / COORD_STEP).astype(np.int64) * 4_000_000_000 \
        + np.round(trees["y"] / COORD_STEP).astype(np.int64)
    _, first = np.unique(key, return_index=True)
    if len(first) == _count(trees):
        return trees
    return _take(trees, np.sort(first))


# --------------------------------------------------------------------------
# Output tables
# --------------------------------------------------------------------------

def _to_dataframe(trees):
    species = trees["species"].astype(np.intp)
    dbh = np.round(trees["dbh"], 2)
    height = np.round(trees["height"], 2)
    itd = trees["itd"]
    return pd.DataFrame({
        "X": trees["x"],
        "Y": trees["y"],
        "H": height,
        "DBH": dbh,
        "Species": np.array(SPECIES_CODES, dtype=object)[species],
        "Latvus_h": np.where(itd, np.round(trees["crown_l"], 2), np.nan),
        "Latvus_d": np.where(itd, np.round(trees["crown_d"], 2), np.nan),
        "Status": np.where(trees["dead"], "Dead", "Alive").astype(object),
        "Type": np.where(itd, "ITD", "Simulated").astype(object),
        "V": np.round(stem_volume(dbh, height, species), 4),
    }, columns=list(COLUMNS))


def _truth(trees, info):
    table = np.array(info, dtype=object).reshape(-1, len(TRUTH_COLUMNS))
    rows = table[trees["layer"]] if len(info) else np.empty((0, len(TRUTH_COLUMNS)), object)
    return pd.DataFrame(rows, columns=list(TRUTH_COLUMNS))


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------

def _layer_label(layer):
    return "background" if layer["id"] == BACKGROUND_ID else f'layer "{layer["name"]}" ({layer["id"]})'


def _active_layers(recipe):
    if recipe["background"]["enabled"]:
        yield dict(recipe["background"], id=BACKGROUND_ID, name="Background",
                   shape={"kind": "window"})
    for layer in recipe["layers"]:
        if layer["enabled"]:
            yield layer


def generate(recipe, seed=None, density_scale=1.0):
    """Generate the forest of a recipe.

    ``seed`` overrides the recipe's seed; ``density_scale`` < 1 gives a quick
    draft with proportionally fewer trees.  Raises RecipeError for an
    invalid recipe.  Returns a Forest whose ``recipe`` records the seed used.
    """
    start = time.perf_counter()
    recipe = validate_recipe(recipe)
    if seed is not None:
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or not 0 <= seed < 2 ** 63:
            raise RecipeError(f"seed: must be a whole number >= 0, got {seed!r}.")
        recipe["seed"] = int(seed)
    seed = recipe["seed"]
    if not 0 < density_scale <= 1:
        raise ValueError("density_scale must be in (0, 1]")
    window = window_geometry(recipe["window"])
    laser = recipe["laser_artefacts"]
    lattice = lattice_offset(seed)
    trees, info, warnings = _empty(), [], []

    for layer in _active_layers(recipe):
        index = len(info)
        info.append((layer["id"], layer["name"], layer["type"], layer.get("preset")))
        geom = shape_geometry(layer["shape"], window)
        if geom.is_empty:
            warnings.append(f"{_layer_label(layer)}: the shape lies outside the window; skipped.")
            continue
        rng = _stream(seed, 1, layer_stream_key(layer["id"]))
        resolved = resolve_layer(layer)
        if resolved.type in ADDING_TYPES:
            if layer.get("mode") == "replace":
                trees = _take(trees, ~points_touching(geom, trees["x"], trees["y"]))
            new, note = _adding_trees(resolved, geom, rng, laser, lattice, density_scale)
            new["layer"] = np.full(_count(new), index, np.int32)
            trees = _concat(trees, new)
            if note:
                warnings.append(f"{_layer_label(layer)}: {note}.")
        elif resolved.type == "clear":
            trees = _take(trees, ~points_touching(geom, trees["x"], trees["y"]))
        elif resolved.type == "thin":
            trees = _take(trees, ~_thin_mask(trees, geom, resolved.params, rng))

    trees = _drop_duplicate_positions(trees)
    counts = np.bincount(trees["layer"], minlength=len(info))
    return Forest(
        trees=_to_dataframe(trees),
        truth=_truth(trees, info),
        recipe=recipe,
        window=window,
        warnings=warnings,
        layer_counts={ident: int(n) for (ident, *_), n in zip(info, counts)},
        elapsed=time.perf_counter() - start,
        density_scale=density_scale,
    )
