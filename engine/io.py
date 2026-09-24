"""Reading and writing tree tables: CSV, RDS, truth file, recipe copy, report.

The tree table always has the columns of ``COLUMNS`` in that order.  CSV
files use "NA" for missing values, 2 decimals for coordinates, sizes and
crowns and 4 decimals for volume.  RDS files are written with pyreadr as a
data.frame whose string columns are ``character`` in R and whose missing
values are R's ``NA`` (not ``NaN``).
"""

import os
import threading

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon

from .generate import COLUMNS, TRUTH_COLUMNS
from .geometry import window_geometry
from .recipe import RecipeError, atomic_write_text, load_recipe, recipe_to_json, validate_recipe

STRING_COLUMNS = ("Species", "Status", "Type")
FLOAT_COLUMNS = ("X", "Y", "H", "DBH", "Latvus_h", "Latvus_d", "V")
DECIMALS = {"X": 2, "Y": 2, "H": 2, "DBH": 2, "Latvus_h": 2, "Latvus_d": 2, "V": 4}
NA = "NA"
# R's NA_real_ is a NaN with payload 1954; pyreadr writes the bits unchanged.
R_NA_REAL = np.array([0x7FF00000000007A2], dtype=np.uint64).view(np.float64)[0]
TABLE_FORMATS = ("csv", "rds")


class TreeFileError(ValueError):
    """A tree file could not be read or does not look like a tree table."""


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def _format_column(values, decimals):
    text = np.char.mod(f"%.{decimals}f", np.nan_to_num(values))
    return np.where(np.isnan(values), NA, text)


def trees_to_csv_text(trees):
    """The CSV text of a tree table (header plus one line per tree)."""
    cols = []
    for name in COLUMNS:
        values = trees[name].to_numpy()
        if name in DECIMALS:
            cols.append(_format_column(values.astype(float), DECIMALS[name]))
        else:
            cols.append(np.asarray(values, dtype=str))
    lines = [",".join(COLUMNS)]
    if len(trees):
        lines.extend(_join_rows(cols))
    return "\n".join(lines) + "\n"


def _join_rows(cols):
    out = cols[0].astype(object)
    for col in cols[1:]:
        out = out + "," + col.astype(object)
    return out.tolist()


def write_csv(trees, path):
    """Write the tree table as CSV (missing values as NA)."""
    atomic_write_text(path, trees_to_csv_text(trees))


def table_for_r(trees):
    """A copy prepared for pyreadr: object string columns, NaN replaced by R's NA."""
    out = pd.DataFrame({name: trees[name].to_numpy() for name in COLUMNS})
    for name in STRING_COLUMNS:
        out[name] = pd.Series(np.asarray(trees[name].to_numpy(), dtype=str), dtype=object)
    for name in FLOAT_COLUMNS:
        values = trees[name].to_numpy(dtype=float).copy()
        values[np.isnan(values)] = R_NA_REAL
        out[name] = values
    return out


def write_rds(trees, path):
    """Write the tree table as an R data.frame with pyreadr."""
    import pyreadr
    tmp = path + ".part"
    try:
        pyreadr.write_rds(tmp, table_for_r(trees), compress="gzip")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_truth(truth, path):
    """Write the per-tree ground truth (layer id, name, type, preset) as CSV."""
    text = truth.astype(object).where(truth.notna(), NA).to_csv(index=False, lineterminator="\n")
    atomic_write_text(path, text)


def output_paths(base):
    """File names written for an output base name such as ``out/forest``."""
    return {
        "csv": base + ".csv",
        "rds": base + ".rds",
        "truth": base + "-truth.csv",
        "recipe": base + "-recipe.json",
        "report": base + "-report.txt",
    }


def write_forest(forest, base, formats=("csv",), report_text=None, truth=True, recipe=True):
    """Write the tree table in ``formats`` plus truth, recipe copy and report.

    The truth file and recipe copy are written unless switched off; the
    report only when ``report_text`` is given.  Returns the list of written
    paths.  Image formats are handled by the caller (the engine does not
    import matplotlib).
    """
    folder = os.path.dirname(os.path.abspath(base))
    os.makedirs(folder, exist_ok=True)
    paths = output_paths(base)
    written = []
    for fmt in formats:
        if fmt == "csv":
            write_csv(forest.trees, paths["csv"])
        elif fmt == "rds":
            write_rds(forest.trees, paths["rds"])
        else:
            raise ValueError(f"unknown table format {fmt!r}")
        written.append(paths[fmt])
    if truth:
        write_truth(forest.truth, paths["truth"])
        written.append(paths["truth"])
    if recipe:
        atomic_write_text(paths["recipe"], recipe_to_json(forest.recipe))
        written.append(paths["recipe"])
    if report_text is not None:
        atomic_write_text(paths["report"], report_text)
        written.append(paths["report"])
    return written


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def _species_text(values):
    """Species codes as text: 1.0 -> "1", factors and numbers become strings."""
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(NA)
        elif isinstance(v, (float, np.floating)) and float(v).is_integer():
            out.append(str(int(v)))
        else:
            out.append(str(v))
    return np.array(out, dtype=object)


def normalise_table(df, source="table"):
    """Bring a table read from disk into the output format.

    Column names are matched case-insensitively.  X and Y are required;
    other missing columns are filled with NA (floats) or "NA" (text).
    """
    lookup = {str(c).lower(): c for c in df.columns}
    missing = [c for c in ("X", "Y") if c.lower() not in lookup]
    if missing:
        raise TreeFileError(f"{source}: no column {' or '.join(missing)}; found "
                            + ", ".join(map(str, df.columns[:12])) + ".")
    out = {}
    for name in COLUMNS:
        col = lookup.get(name.lower())
        if name in STRING_COLUMNS:
            if col is None:
                out[name] = np.full(len(df), NA, dtype=object)
            elif name == "Species":
                out[name] = _species_text(df[col].astype(object).to_numpy())
            else:
                values = df[col].astype(object).to_numpy()
                out[name] = np.array([NA if (v is None or (isinstance(v, float) and np.isnan(v)))
                                      else str(v) for v in values], dtype=object)
        else:
            if col is None:
                out[name] = np.full(len(df), np.nan)
            else:
                out[name] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    table = pd.DataFrame(out, columns=list(COLUMNS))
    bad = ~(np.isfinite(table["X"]) & np.isfinite(table["Y"]))
    if bad.all() and len(table):
        raise TreeFileError(f"{source}: X and Y contain no numbers.")
    return table[~bad].reset_index(drop=True)


def read_trees(path):
    """Read a CSV or RDS tree table; returns a normalised DataFrame."""
    name = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".rds":
            import pyreadr
            result = pyreadr.read_r(path)
            if not result:
                raise TreeFileError(f"{name}: the RDS file holds no data frame.")
            df = next(iter(result.values()))
        elif ext in (".csv", ".txt"):
            df = pd.read_csv(path, dtype={c: str for c in STRING_COLUMNS},
                             na_values=[NA], keep_default_na=True, sep=None, engine="python")
        else:
            raise TreeFileError(f"{name}: unknown file type; open a .csv or .rds file.")
    except TreeFileError:
        raise
    except FileNotFoundError:
        raise TreeFileError(f"{name}: file not found.") from None
    except Exception as exc:         # parsers raise many different errors
        raise TreeFileError(f"{name}: cannot read the file ({exc}).") from None
    return normalise_table(df, source=name)


def read_truth(path, n_trees):
    """Read a truth file if it matches ``n_trees`` rows; otherwise return None."""
    try:
        truth = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return None
    if len(truth) != n_trees or "layer_id" not in truth.columns:
        return None
    return truth.reindex(columns=list(TRUTH_COLUMNS))


def sidecar_paths(tree_path):
    """Truth and recipe files that belong to a tree file, if named by convention."""
    base = os.path.splitext(tree_path)[0]
    return {"truth": base + "-truth.csv", "recipe": base + "-recipe.json"}


def fit_window(trees, margin=0.5):
    """A window spec that contains every tree, strictly inside.

    If the trees fill their minimum bounding circle well (convex hull at
    least 90 % of the circle, as in the circular treemaps), the window is
    that circle; otherwise it is the convex hull widened by 1 m.  A margin
    keeps trees on the edge inside the window.
    """
    xy = np.column_stack([trees["X"].to_numpy(float), trees["Y"].to_numpy(float)])
    if len(xy) < 3:
        cx, cy = (xy.mean(axis=0) if len(xy) else (0.0, 0.0))
        return {"kind": "circle", "center": [round(float(cx), 2), round(float(cy), 2)], "radius": 10.0}
    points = shapely.multipoints(xy)
    hull = points.convex_hull
    circle = shapely.minimum_bounding_circle(points)
    if isinstance(hull, Polygon) and circle.area > 0 and hull.area / circle.area >= 0.9:
        centre = circle.centroid
        cx, cy = round(centre.x, 2), round(centre.y, 2)
        radius = float(np.max(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)))
        # the window circle is drawn as a 256-gon, slightly inside the true circle
        radius = np.ceil((radius * 1.0001 + margin) * 100) / 100
        return {"kind": "circle", "center": [cx, cy], "radius": float(radius)}
    outline = hull.buffer(1.0 + margin, quad_segs=2).simplify(0.25 * margin)
    vertices = [[round(x, 2), round(y, 2)] for x, y in list(outline.exterior.coords)[:-1]]
    return {"kind": "polygon", "vertices": vertices}


def window_for_trees(tree_path, trees):
    """The area a tree file covers.

    Uses the window of a matching ``<name>-recipe.json`` when present,
    otherwise a window fitted to the trees (see ``fit_window``).  Returns
    (geometry, window spec, where it came from, recipe or None).
    """
    recipe_path = sidecar_paths(tree_path)["recipe"]
    if os.path.exists(recipe_path):
        try:
            recipe = load_recipe(recipe_path)
            return (window_geometry(recipe["window"]), recipe["window"],
                    "from the recipe file", recipe)
        except RecipeError:
            pass
    spec = fit_window(trees)
    return window_geometry(spec), spec, "fitted to the trees", None


# --------------------------------------------------------------------------
# Tree files as the background of a recipe
# --------------------------------------------------------------------------

_BASE_CACHE = {}
_BASE_LOCK = threading.Lock()


def resolve_path(path, base_dir=None):
    """Absolute path of ``path``; relative paths are taken from ``base_dir`` (or the cwd)."""
    path = os.path.expanduser(path)
    if not os.path.isabs(path) and base_dir:
        path = os.path.join(base_dir, path)
    return os.path.abspath(path)


def load_base_trees(path):
    """Read a tree file used as a background, cached while the file is unchanged.

    Raises RecipeError naming the background's ``file`` key if the file
    cannot be read.  The returned table must not be modified.
    """
    try:
        stat = os.stat(path)
    except OSError as exc:
        raise RecipeError(f'background, key "file": cannot read {path} '
                          f"({exc.strerror}).") from None
    stamp = (stat.st_mtime_ns, stat.st_size)
    with _BASE_LOCK:
        cached = _BASE_CACHE.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
    try:
        table = read_trees(path)
    except TreeFileError as exc:
        raise RecipeError(f'background, key "file": {exc}') from None
    with _BASE_LOCK:
        if len(_BASE_CACHE) >= 4:
            _BASE_CACHE.pop(next(iter(_BASE_CACHE)))
        _BASE_CACHE[path] = (stamp, table)
    return table


def infer_lattice_offset(trees, step=0.5, agreement=0.9):
    """The offset of the laser lattice of a table's ITD trees, or None.

    Returns (ox, oy) if at least ``agreement`` of the ITD coordinates share
    the same position modulo ``step`` (rounded to 0.01 m).
    """
    itd = (trees["Type"].astype(str) == "ITD").to_numpy()
    if itd.sum() < 20:
        return None
    offset = []
    for column in ("X", "Y"):
        cents = np.round(np.mod(trees[column].to_numpy(float)[itd], step) * 100).astype(np.int64)
        cents %= int(round(step * 100))
        values, counts = np.unique(cents, return_counts=True)
        if counts.max() < agreement * itd.sum():
            return None
        offset.append(float(values[np.argmax(counts)]) / 100)
    return tuple(offset)


def recipe_for_file(path, trees):
    """A new recipe whose background is the tree file at ``path`` (for painting on it).

    The window comes from a matching ``<name>-recipe.json`` if there is one,
    otherwise it is fitted to the trees.
    """
    recipe_path = sidecar_paths(path)["recipe"]
    window, laser = None, True
    if os.path.exists(recipe_path):
        try:
            sidecar = load_recipe(recipe_path)
            window, laser = sidecar["window"], sidecar["laser_artefacts"]
        except RecipeError:
            pass
    name = os.path.splitext(os.path.basename(path))[0] + "_edited"
    return validate_recipe({
        "name": name,
        "description": f"Trees from {os.path.basename(path)} with painted layers.",
        "window": window or fit_window(trees),
        "laser_artefacts": laser,
        "background": {"file": os.path.abspath(path)},
        "layers": [],
    })
