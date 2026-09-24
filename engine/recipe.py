"""Recipes: load, validate, normalise, resolve and save (JSON).

A recipe is a plain JSON-compatible dict.  ``validate_recipe`` checks every
key, collects all problems (each naming the layer and the key) and returns
a normalised deep copy; ``resolve_layer`` merges type defaults, the preset
and the layer's own keys into the values used for generation.
"""

import copy
import difflib
import json
import os
import re
import zlib
from typing import NamedTuple

import numpy as np

from .attributes import Composition, lognormal_mu
from .geometry import (
    SHAPE_KINDS,
    WINDOW_KINDS,
    area_ha,
    raw_shape,
    shape_geometry,
    window_geometry,
)
from .presets import (
    DEFAULT_BACKGROUND,
    DEFAULT_DBH_SIGMA,
    DEFAULT_DEAD_SHARE,
    DEFAULT_MEDIAN_DBH,
    DEFAULT_SHARES,
    DEFAULT_WINDOW,
    PRESETS,
    SIZE_CLASSES,
    SPECIES,
    TYPE_DEFAULTS,
)
from .processes import max_regular_density, practical_regular_density

FORMAT = "synthforest-recipe"
VERSION = 1
ADDING_TYPES = ("random", "clustered", "regular", "rows", "gradient")
REMOVING_TYPES = ("clear", "thin")
LAYER_TYPES = ADDING_TYPES + REMOVING_TYPES
MODES = ("add", "replace")
MAX_TREES = 1_000_000
DEFAULT_SEED = 1

TOP_KEYS = ("format", "version", "name", "description", "seed", "window",
            "laser_artefacts", "background", "layers")
LAYER_KEYS = ("id", "name", "enabled", "visible", "type", "mode", "preset",
              "shape", "params", "composition")
BACKGROUND_KEYS = ("enabled", "type", "preset", "params", "composition", "file")
COMPOSITION_KEYS = ("species", "size_class", "dbh", "dead_share")

# (minimum, maximum, unit) of every numeric parameter.
PARAM_RANGES = {
    "random": {"density": (0, 20000, "trees/ha")},
    "clustered": {"cluster_density": (0, 2000, "clusters/ha"),
                  "trees_per_cluster": (0, 1000, "trees"),
                  "cluster_radius": (0.1, 200, "m")},
    "regular": {"density": (0, 20000, "trees/ha"), "min_distance": (0, 50, "m")},
    "rows": {"row_spacing": (1, 100, "m"), "tree_spacing": (1, 100, "m"),
             "angle": (-360, 360, "deg"), "jitter": (0, 10, "m")},
    "gradient": {"density_start": (0, 20000, "trees/ha"),
                 "density_end": (0, 20000, "trees/ha"),
                 "direction": (-360, 360, "deg")},
    "clear": {},
    "thin": {"fraction": (0, 1, ""), "dbh_below": (4.5, 150, "cm"),
             "dbh_above": (4.5, 150, "cm")},
}
SHAPE_KEYS = {
    "window": (),
    "circle": ("center", "radius"),
    "rectangle": ("center", "width", "height"),
    "polygon": ("vertices",),
    "corridor": ("points", "width"),
}
COORD_LIMIT = 1e7


class RecipeError(ValueError):
    """A recipe problem; ``errors`` lists every message."""

    def __init__(self, errors):
        self.errors = list(errors) if not isinstance(errors, str) else [errors]
        super().__init__("\n".join(self.errors))


class ResolvedLayer(NamedTuple):
    """Everything generation needs for one layer."""

    type: str
    params: dict
    composition: object      # Composition, or None for removing layers


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


class _Checker:
    """Collects validation errors."""

    def __init__(self):
        self.errors = []

    def error(self, where, key, message, suggestion=None):
        text = f"{where}, key \"{key}\": {message}" if key else f"{where}: {message}"
        if suggestion:
            text += f" Suggestion: {suggestion}"
        self.errors.append(text)

    def mapping(self, where, key, value):
        if not isinstance(value, dict):
            self.error(where, key, f"must be an object {{...}}, got {type(value).__name__}.")
            return None
        return value

    def unknown_keys(self, where, obj, allowed, prefix=""):
        for key in obj:
            if key not in allowed:
                close = difflib.get_close_matches(str(key), allowed, n=1)
                hint = f"did you mean \"{prefix}{close[0]}\"?" if close else (
                    "allowed keys are " + ", ".join(f'"{prefix}{k}"' for k in allowed) + "."
                    if allowed else "this object takes no keys.")
                self.error(where, prefix + str(key), "unknown key.", hint)

    def number(self, where, key, value, lo=None, hi=None, unit=""):
        if not _is_number(value):
            self.error(where, key, f"must be a number, got {value!r}.")
            return None
        unit = f" {unit}" if unit else ""
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            self.error(where, key, f"must be between {lo:g} and {hi:g}{unit}, got {value:g}.",
                       f"use a value in {lo:g}..{hi:g}{unit}.")
            return None
        return float(value)

    def boolean(self, where, key, value):
        if not isinstance(value, bool):
            self.error(where, key, f"must be true or false, got {value!r}.")
            return None
        return value

    def choice(self, where, key, value, options):
        if value not in options:
            close = difflib.get_close_matches(str(value), options, n=1)
            hint = f"did you mean \"{close[0]}\"?" if close else "choose one of " + ", ".join(options) + "."
            self.error(where, key, f"unknown value {value!r}.", hint)
            return None
        return value

    def point(self, where, key, value):
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or not all(_is_number(v) and abs(v) < COORD_LIMIT for v in value)):
            self.error(where, key, f"must be a point [x, y] in metres, got {value!r}.")
            return None
        return [float(value[0]), float(value[1])]

    def points(self, where, key, value, minimum):
        if not isinstance(value, (list, tuple)) or len(value) < minimum:
            self.error(where, key, f"must be a list of at least {minimum} points [[x, y], ...].")
            return None
        out = [self.point(where, f"{key}[{i}]", p) for i, p in enumerate(value)]
        return None if any(p is None for p in out) else out


# --------------------------------------------------------------------------
# Validation of the parts
# --------------------------------------------------------------------------

def _shape(c, where, key, spec, kinds):
    def k(field):
        return f"{key}.{field}" if key else field

    spec = c.mapping(where, key, spec)
    if spec is None:
        return None
    kind = c.choice(where, k("kind"), spec.get("kind"), kinds)
    if kind is None:
        return None
    c.unknown_keys(where, spec, ("kind",) + SHAPE_KEYS[kind], prefix=k(""))
    out = {"kind": kind}
    for field in SHAPE_KEYS[kind]:
        if field not in spec:
            c.error(where, k(field), f"missing (a {kind} needs "
                    + ", ".join(SHAPE_KEYS[kind]) + ").")
            return None
    if kind in ("circle", "rectangle"):
        out["center"] = c.point(where, k("center"), spec["center"])
    if kind == "circle":
        out["radius"] = c.number(where, k("radius"), spec["radius"], 0.01, 1e5, "m")
    if kind == "rectangle":
        out["width"] = c.number(where, k("width"), spec["width"], 0.01, 2e5, "m")
        out["height"] = c.number(where, k("height"), spec["height"], 0.01, 2e5, "m")
    if kind == "polygon":
        out["vertices"] = c.points(where, k("vertices"), spec["vertices"], 3)
    if kind == "corridor":
        out["points"] = c.points(where, k("points"), spec["points"], 1)
        out["width"] = c.number(where, k("width"), spec["width"], 0.01, 1e4, "m")
    if any(v is None for v in out.values()):
        return None
    if kind == "polygon" and raw_shape(out).area <= 0:
        c.error(where, k("vertices"), "the polygon has no area.",
                "give at least three points that are not on one line.")
        return None
    return out


def _composition(c, where, comp):
    comp = c.mapping(where, "composition", comp)
    if comp is None:
        return None
    c.unknown_keys(where, comp, COMPOSITION_KEYS, prefix="composition.")
    out = {}
    if "species" in comp:
        shares = c.mapping(where, "composition.species", comp["species"])
        if shares is not None:
            c.unknown_keys(where, shares, SPECIES, prefix="composition.species.")
            vals = {k: c.number(where, f"composition.species.{k}", v, 0, 1)
                    for k, v in shares.items() if k in SPECIES}
            if all(v is not None for v in vals.values()):
                total = sum(vals.values())
                if abs(total - 1.0) > 0.01:
                    fix = ", ".join(f"{k} {v / total:.3f}" for k, v in vals.items()) if total > 0 \
                        else "e.g. pine 0.27, spruce 0.43, broadleaved 0.30"
                    c.error(where, "composition.species",
                            f"shares must sum to 1, they sum to {total:.3f}.",
                            f"scale them: {fix}.")
                else:
                    out["species"] = vals
    if "size_class" in comp and "dbh" in comp:
        c.error(where, "composition.dbh", "use either size_class or dbh, not both.",
                "remove one of them.")
    elif "size_class" in comp:
        size = c.choice(where, "composition.size_class", comp["size_class"], list(SIZE_CLASSES))
        if size is not None:
            out["size_class"] = size
    elif "dbh" in comp:
        dbh = c.mapping(where, "composition.dbh", comp["dbh"])
        if dbh is not None:
            c.unknown_keys(where, dbh, ("median", "sigma"), prefix="composition.dbh.")
            median = c.number(where, "composition.dbh.median", dbh.get("median"), 5, 100, "cm")
            sigma = c.number(where, "composition.dbh.sigma", dbh.get("sigma"), 0.05, 1.5)
            if median is not None and sigma is not None:
                try:
                    lognormal_mu(median, sigma)
                    out["dbh"] = {"median": median, "sigma": sigma}
                except ValueError as exc:
                    c.error(where, "composition.dbh", str(exc) + ".",
                            "raise the median or lower sigma.")
    if "dead_share" in comp and comp["dead_share"] is not None:
        dead = c.number(where, "composition.dead_share", comp["dead_share"], 0, 1)
        if dead is not None:
            out["dead_share"] = dead
    return out


def _params(c, where, ltype, params):
    params = c.mapping(where, "params", params)
    if params is None:
        return None
    ranges = PARAM_RANGES[ltype]
    c.unknown_keys(where, params, tuple(ranges), prefix="params.")
    out = {}
    for key, value in params.items():
        if key in ranges:
            lo, hi, unit = ranges[key]
            if ltype == "thin" and key in ("dbh_below", "dbh_above") and value is None:
                continue
            number = c.number(where, f"params.{key}", value, lo, hi, unit)
            if number is not None:
                out[key] = number
    return out


def _pattern_part(c, where, obj, allow_types):
    """Validate type, preset, params and composition shared by layers and background."""
    out = {}
    preset = obj.get("preset")
    if preset is not None:
        preset = c.choice(where, "preset", preset, list(PRESETS))
        if preset is not None:
            out["preset"] = preset
    ltype = obj.get("type")
    if ltype is None:
        if preset is None:
            c.error(where, "type", "missing.", "set a type (" + ", ".join(allow_types)
                    + ") or a preset.")
            return None
        ltype = PRESETS[preset]["type"]
    ltype = c.choice(where, "type", ltype, list(allow_types))
    if ltype is None:
        return None
    out["type"] = ltype
    params = _params(c, where, ltype, obj.get("params", {}))
    if params is not None:
        out["params"] = params
    if ltype in ADDING_TYPES:
        comp = _composition(c, where, obj.get("composition", {}))
        if comp is not None:
            out["composition"] = comp
    else:
        for key in ("preset", "composition", "mode"):
            if key in obj:
                c.error(where, key, f"not used by type \"{ltype}\".", f"remove \"{key}\".")
    return out


def _check_resolved(c, where, layer):
    """Cross-key checks that need the merged parameters."""
    resolved = resolve_layer(layer)
    p = resolved.params
    if resolved.type == "regular" and p["min_distance"] > 0:
        limit = max_regular_density(p["min_distance"])
        if p["density"] > limit:
            practical = practical_regular_density(p["min_distance"])
            best_r = np.sqrt(0.547 / (np.pi / 4) / (p["density"] / 10_000))
            c.error(where, "params.density",
                    f"{p['density']:g} trees/ha is impossible with min_distance "
                    f"{p['min_distance']:g} m (hexagonal packing allows {limit:.0f}).",
                    f"use density <= {practical:.0f} trees/ha or min_distance <= {best_r:.2f} m.")
    if resolved.type == "thin":
        below, above = p.get("dbh_below"), p.get("dbh_above")
        if below is not None and above is not None and above >= below:
            c.error(where, "params.dbh_above",
                    f"dbh_above ({above:g}) must be smaller than dbh_below ({below:g}).",
                    "thin trees with dbh_above < dbh < dbh_below, or set only one of them.")


def _layer_where(index, layer):
    name = layer.get("name") if isinstance(layer, dict) else None
    ident = layer.get("id") if isinstance(layer, dict) else None
    label = f'layer {index + 1}'
    if name:
        label += f' "{name}"'
    if ident:
        label += f" ({ident})"
    return label


def _layer(c, index, layer, seen, taken):
    where = _layer_where(index, layer)
    layer = c.mapping(where, None, layer)
    if layer is None:
        return None
    c.unknown_keys(where, layer, LAYER_KEYS)
    out = {}
    ident = layer.get("id")
    if ident is None:
        k = 1
        while f"L{k}" in taken or f"L{k}" in seen:
            k += 1
        ident = f"L{k}"
    elif not isinstance(ident, str) or not ident.strip():
        c.error(where, "id", f"must be a non-empty text, got {ident!r}.")
        return None
    if ident in seen:
        c.error(where, "id", f"\"{ident}\" is used by another layer.", "give every layer its own id.")
    seen.add(ident)
    out["id"] = ident
    pattern = _pattern_part(c, where, layer, LAYER_TYPES)
    if pattern is None:
        return None
    name = layer.get("name", pattern.get("preset") or pattern["type"])
    if not isinstance(name, str):
        c.error(where, "name", f"must be text, got {name!r}.")
    out["name"] = str(name)
    for key in ("enabled", "visible"):
        value = c.boolean(where, key, layer.get(key, True))
        out[key] = True if value is None else value
    if pattern["type"] in ADDING_TYPES:
        mode = c.choice(where, "mode", layer.get("mode", "add"), list(MODES))
        out["mode"] = mode or "add"
    out.update(pattern)
    if "shape" not in layer:
        c.error(where, "shape", "missing.", 'e.g. {"kind": "window"} for the whole area.')
        return None
    shape = _shape(c, where, "shape", layer["shape"], SHAPE_KINDS)
    if shape is None:
        return None
    out["shape"] = shape
    return out


def _background(c, background):
    where = "background"
    if background is None:
        background = {"enabled": False, "type": "random"}
    background = c.mapping(where, None, background)
    if background is None:
        return None
    c.unknown_keys(where, background, BACKGROUND_KEYS)
    out = {}
    value = c.boolean(where, "enabled", background.get("enabled", True))
    out["enabled"] = True if value is None else value
    if "file" in background:
        return _file_background(c, where, background, out)
    pattern = _pattern_part(c, where, background, ADDING_TYPES)
    if pattern is None:
        return None
    out.update(pattern)
    return out


def _file_background(c, where, background, out):
    """A background of trees read from a CSV or RDS file (the file is read at generation)."""
    path = background["file"]
    if not isinstance(path, str) or not path.strip():
        c.error(where, "file", f"must be the path of a CSV or RDS tree file, got {path!r}.")
        return None
    if not path.lower().endswith((".csv", ".txt", ".rds")):
        c.error(where, "file", f"{path!r} is not a .csv or .rds file.")
        return None
    for key in ("type", "preset", "params", "composition"):
        if key in background:
            c.error(where, key, "not used when the background comes from a file.",
                    f"remove \"{key}\", or remove \"file\" to generate the background.")
    out["file"] = path
    return out


def is_file_background(background):
    """True if the (validated) background is read from a tree file."""
    return "file" in background


def validate_recipe(recipe):
    """Validate a recipe and return a normalised deep copy.

    Raises RecipeError listing every problem.  Normalisation fills in the
    structural defaults (window, seed, background, ids, names, flags, mode)
    but leaves pattern parameters sparse so presets keep working.
    """
    c = _Checker()
    if not isinstance(recipe, dict):
        raise RecipeError("recipe: must be a JSON object {...}.")
    c.unknown_keys("recipe", recipe, TOP_KEYS)
    if recipe.get("format", FORMAT) != FORMAT:
        c.error("recipe", "format", f"must be \"{FORMAT}\", got {recipe.get('format')!r}.")
    if recipe.get("version", VERSION) != VERSION:
        c.error("recipe", "version", f"version {recipe.get('version')!r} is not supported "
                f"(this program reads version {VERSION}).")
    out = {"format": FORMAT, "version": VERSION}
    name = recipe.get("name", "forest")
    if not isinstance(name, str) or not name.strip():
        c.error("recipe", "name", f"must be non-empty text, got {name!r}.")
    out["name"] = str(name)
    if "description" in recipe:
        if not isinstance(recipe["description"], str):
            c.error("recipe", "description", "must be text.")
        out["description"] = str(recipe["description"])
    seed = recipe.get("seed", DEFAULT_SEED)
    if seed is None:
        seed = DEFAULT_SEED
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 63:
        c.error("recipe", "seed", f"must be a whole number >= 0, got {seed!r}.")
    out["seed"] = seed
    out["window"] = _shape(c, "window", None, recipe.get("window", DEFAULT_WINDOW), WINDOW_KINDS)
    laser = c.boolean("recipe", "laser_artefacts", recipe.get("laser_artefacts", True))
    out["laser_artefacts"] = True if laser is None else laser
    out["background"] = _background(c, copy.deepcopy(recipe.get("background", DEFAULT_BACKGROUND)))
    layers = recipe.get("layers", [])
    if not isinstance(layers, list):
        c.error("recipe", "layers", "must be a list [...] of layers.")
        layers = []
    taken = {layer["id"] for layer in layers
             if isinstance(layer, dict) and isinstance(layer.get("id"), str)}
    seen = set()
    out["layers"] = []
    for i, layer in enumerate(layers):
        norm = _layer(c, i, layer, seen, taken)
        if norm is not None:
            out["layers"].append(norm)
    if c.errors:
        raise RecipeError(c.errors)
    if out["background"]["enabled"] and not is_file_background(out["background"]):
        _check_resolved(c, "background", out["background"])
    for i, layer in enumerate(out["layers"]):
        _check_resolved(c, _layer_where(i, layer), layer)
    if not c.errors:
        _check_tree_count(c, out)
    if c.errors:
        raise RecipeError(c.errors)
    return out


def expected_trees(layer, geom):
    """Rough expected number of trees an adding layer creates in ``geom``."""
    resolved = resolve_layer(layer)
    p, kind = resolved.params, resolved.type
    if kind in ("random", "regular"):
        density = p["density"]
    elif kind == "clustered":
        density = p["cluster_density"] * p["trees_per_cluster"]
    elif kind == "gradient":
        density = max(p["density_start"], p["density_end"])
    elif kind == "rows":
        density = 10_000.0 / (p["row_spacing"] * p["tree_spacing"])
    else:
        return 0.0
    return density * area_ha(geom)


def _check_tree_count(c, recipe):
    try:
        window = window_geometry(recipe["window"])
    except Exception as exc:        # shapely may reject pathological input
        c.error("window", None, f"cannot build the window: {exc}.")
        return
    if window.area <= 0:
        c.error("window", None, "the window has no area.")
        return
    total, parts = 0.0, []
    if recipe["background"]["enabled"] and not is_file_background(recipe["background"]):
        n = expected_trees(recipe["background"], window)
        total += n
        parts.append((n, "background"))
    for i, layer in enumerate(recipe["layers"]):
        if layer["enabled"] and layer["type"] in ADDING_TYPES:
            n = expected_trees(layer, shape_geometry(layer["shape"], window))
            total += n
            parts.append((n, _layer_where(i, layer)))
    if total > MAX_TREES:
        worst = ", ".join(f"{label} (~{n:,.0f})" for n, label in sorted(parts, reverse=True)[:3])
        c.error("recipe", None, f"it would create about {total:,.0f} trees; the limit is "
                f"{MAX_TREES:,}.", f"lower the density of the largest layers: {worst}.")


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def _resolve_composition(comp):
    if "species" in comp:
        raw = np.array([comp["species"].get(s, 0.0) for s in SPECIES], dtype=float)
        shares = tuple(raw / raw.sum())
    else:
        shares = DEFAULT_SHARES
    if "size_class" in comp:
        dbh = (SIZE_CLASSES[comp["size_class"]],) * 3
    elif "dbh" in comp:
        dbh = ((comp["dbh"]["median"], comp["dbh"]["sigma"]),) * 3
    else:
        dbh = tuple((m, DEFAULT_DBH_SIGMA) for m in DEFAULT_MEDIAN_DBH)
    if comp.get("dead_share") is not None:
        dead = (comp["dead_share"],) * 3
    else:
        dead = DEFAULT_DEAD_SHARE
    return Composition(shares=shares, dbh=dbh, dead=dead)


def merged_composition(layer):
    """The layer's composition dict with preset values filled in."""
    preset = PRESETS.get(layer.get("preset"), {})
    merged = copy.deepcopy(preset.get("composition", {}))
    own = layer.get("composition", {})
    if "size_class" in own or "dbh" in own:
        merged.pop("size_class", None)
        merged.pop("dbh", None)
    merged.update(copy.deepcopy(own))
    return merged


def merged_params(layer):
    """Parameters of a (validated) layer: type defaults, then preset, then own keys."""
    preset = PRESETS.get(layer.get("preset"), {})
    ltype = layer["type"]
    params = dict(TYPE_DEFAULTS[ltype])
    if preset.get("type") == ltype:
        params.update(preset["params"])
    params.update(layer.get("params", {}))
    return params


def resolve_layer(layer):
    """Merge defaults, preset and own keys of a validated layer or background."""
    ltype = layer["type"]
    comp = _resolve_composition(merged_composition(layer)) if ltype in ADDING_TYPES else None
    return ResolvedLayer(type=ltype, params=merged_params(layer), composition=comp)


# --------------------------------------------------------------------------
# Files and helpers
# --------------------------------------------------------------------------

def new_recipe(name="forest", seed=DEFAULT_SEED):
    """A fresh default recipe: default window, default background, no layers."""
    return validate_recipe({"name": name, "seed": seed})


def next_layer_id(recipe):
    """A layer id ("L<n>") not used in the recipe."""
    numbers = [int(m.group(1)) for layer in recipe.get("layers", [])
               if isinstance(layer.get("id"), str) and (m := re.fullmatch(r"L(\d+)", layer["id"]))]
    return f"L{max(numbers, default=0) + 1}"


def layer_stream_key(layer_id):
    """Stable integer derived from a layer id (for per-layer random streams)."""
    return zlib.crc32(layer_id.encode("utf-8"))


def recipe_to_json(recipe):
    """Readable JSON text: two-space indents, short number lists on one line."""
    text = json.dumps(recipe, indent=2, ensure_ascii=False)
    return re.sub(r"\[\s+([-+\d.eE]+),\s+([-+\d.eE]+)\s+\]", r"[\1, \2]", text) + "\n"


def parse_recipe(text, source="recipe"):
    """Parse and validate recipe JSON text."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RecipeError(f"{source}: not valid JSON: {exc.msg} at line {exc.lineno}, "
                          f"column {exc.colno}.") from None
    return validate_recipe(data)


def load_recipe(path):
    """Read and validate a recipe file."""
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise RecipeError(f"{path}: cannot read the file ({exc.strerror}).") from None
    return parse_recipe(text, source=os.path.basename(path))


def atomic_write_text(path, text):
    """Write a text file via a temporary file so a crash never leaves half a file."""
    tmp = f"{path}.{os.getpid()}.part"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_recipe(recipe, path):
    """Validate and save a recipe as JSON."""
    atomic_write_text(path, recipe_to_json(validate_recipe(recipe)))
