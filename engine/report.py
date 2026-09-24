"""Realism report: the figures of a tree table next to the default targets."""

import numpy as np

from .attributes import CROWN_DIAMETER_PER_DBH, CROWN_RATIO
from .patterns import clark_evans
from .presets import (
    DEFAULT_DEAD_SHARE,
    DEFAULT_DENSITY,
    DEFAULT_IMPUTED_SHARE,
    DEFAULT_MEDIAN_DBH,
    DEFAULT_MEDIAN_HEIGHT,
    DEFAULT_SHARES,
    SPECIES,
    SPECIES_CODES,
)

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
DEFAULT_TOTAL_DEAD = float(np.dot(DEFAULT_SHARES, DEFAULT_DEAD_SHARE))


def _share(mask):
    return float(np.mean(mask)) if len(mask) else float("nan")


def forest_report(trees, window, window_note=None):
    """Compute the report figures of a tree table inside ``window``.

    Returns a dict; ``format_report`` turns it into text.  Species codes
    other than "1", "2", "3" are counted under "other".
    """
    n = len(trees)
    area_ha = window.area / 10_000.0
    dbh = trees["DBH"].to_numpy(dtype=float)
    height = trees["H"].to_numpy(dtype=float)
    species = trees["Species"].astype(str).to_numpy()
    ba = np.pi * (dbh / 200.0) ** 2
    ba_total = np.nansum(ba)
    rep = {
        "n": n,
        "area_ha": area_ha,
        "window_note": window_note,
        "density": n / area_ha if area_ha > 0 else float("nan"),
        "basal_area": ba_total / area_ha if area_ha > 0 else float("nan"),
        "volume": np.nansum(trees["V"].to_numpy(dtype=float)) / area_ha if area_ha > 0 else float("nan"),
        "dead_share": _share(trees["Status"].to_numpy() == "Dead"),
        "imputed_share": _share(trees["Type"].to_numpy() == "Simulated"),
        "clark_evans": clark_evans(trees["X"].to_numpy(float), trees["Y"].to_numpy(float),
                                   window.area, window.length),
        "species": {},
    }
    other = ~np.isin(species, SPECIES_CODES)
    groups = [(name, code, species == code) for name, code in zip(SPECIES, SPECIES_CODES)]
    if other.any():
        groups.append(("other", "?", other))
    for i, (name, code, mask) in enumerate(groups):
        sub = trees[mask]
        crowns = sub["Latvus_d"].notna().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            crown_d = (sub["Latvus_d"] / sub["DBH"]).to_numpy(float)[crowns]
            crown_l = (sub["Latvus_h"] / sub["H"]).to_numpy(float)[crowns]
        rep["species"][name] = {
            "code": code,
            "n": int(mask.sum()),
            "share_stems": _share(mask),
            "share_ba": float(np.nansum(ba[mask]) / ba_total) if ba_total > 0 else float("nan"),
            "dead_share": _share(sub["Status"].to_numpy() == "Dead"),
            "dbh_q": _quantiles(dbh[mask]),
            "height_q": _quantiles(height[mask]),
            "crown_d_per_dbh": float(np.nanmedian(crown_d)) if crowns.any() else float("nan"),
            "crown_ratio": float(np.nanmedian(crown_l)) if crowns.any() else float("nan"),
            "target": _species_targets(i) if i < 3 else None,
        }
    return rep


def _quantiles(values):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [float("nan")] * len(QUANTILES)
    return [float(q) for q in np.quantile(values, QUANTILES)]


def _species_targets(i):
    return {"share_stems": DEFAULT_SHARES[i], "dead_share": DEFAULT_DEAD_SHARE[i],
            "median_dbh": DEFAULT_MEDIAN_DBH[i], "median_height": DEFAULT_MEDIAN_HEIGHT[i],
            "crown_d_per_dbh": float(CROWN_DIAMETER_PER_DBH[i]), "crown_ratio": float(CROWN_RATIO[i])}


def _fmt(value, digits):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:,.{digits}f}"


def format_report(rep, title="synthforest realism report"):
    """Plain-text report: each figure next to its default target."""
    row = "{:<34}{:>12}{:>12}"
    lines = [title, "=" * len(title)]
    if rep.get("window_note"):
        lines.append(f"Window: {rep['window_note']}")
    lines += ["", row.format("", "value", "default"),
              row.format("Trees", _fmt(rep["n"], 0), ""),
              row.format("Area (ha)", _fmt(rep["area_ha"], 2), ""),
              row.format("Density (trees/ha)", _fmt(rep["density"], 1), _fmt(DEFAULT_DENSITY, 0)),
              row.format("Basal area (m2/ha)", _fmt(rep["basal_area"], 2), ""),
              row.format("Volume (m3/ha)", _fmt(rep["volume"], 1), ""),
              row.format("Dead share", _fmt(rep["dead_share"], 3), _fmt(DEFAULT_TOTAL_DEAD, 3)),
              row.format("Imputed share (Simulated)", _fmt(rep["imputed_share"], 3),
                         _fmt(DEFAULT_IMPUTED_SHARE, 2)),
              row.format("Clark-Evans R", _fmt(rep["clark_evans"], 3), "1 = random"),
              "  (R < 1 clustered, R > 1 regular; Donnelly edge correction)", ""]
    for name, sp in rep["species"].items():
        t = sp["target"] or {}
        lines += [f"{name} (Species \"{sp['code']}\", {sp['n']:,} trees)",
                  row.format("  share by stems", _fmt(sp["share_stems"], 3), _fmt(t.get("share_stems"), 2)),
                  row.format("  share by basal area", _fmt(sp["share_ba"], 3), ""),
                  row.format("  dead share", _fmt(sp["dead_share"], 3), _fmt(t.get("dead_share"), 3)),
                  row.format("  median dbh (cm)", _fmt(sp["dbh_q"][2], 1), _fmt(t.get("median_dbh"), 0)),
                  row.format("  median height (m)", _fmt(sp["height_q"][2], 1), _fmt(t.get("median_height"), 0)),
                  row.format("  crown diameter / dbh (m/cm)", _fmt(sp["crown_d_per_dbh"], 3),
                             _fmt(t.get("crown_d_per_dbh"), 2)),
                  row.format("  crown length / height", _fmt(sp["crown_ratio"], 3), _fmt(t.get("crown_ratio"), 2)),
                  "  dbh quantiles 5/25/50/75/95 %:    " + " ".join(_fmt(q, 1) for q in sp["dbh_q"]),
                  "  height quantiles 5/25/50/75/95 %: " + " ".join(_fmt(q, 1) for q in sp["height_q"]),
                  ""]
    return "\n".join(lines)


def _num(value):
    return f"{value:,.2f}".rstrip("0").rstrip(".").replace(",", " ")


def window_description(spec, hide_centre=False):
    """Short text describing a window spec; ``hide_centre`` leaves out every coordinate."""
    kind = spec["kind"]
    centre = "" if hide_centre else f", centre ({_num(spec['center'][0])}, {_num(spec['center'][1])})" \
        if "center" in spec else ""
    if kind == "circle":
        return f"circle, radius {_num(spec['radius'])} m{centre}"
    if kind == "rectangle":
        return f"rectangle {_num(spec['width'])} x {_num(spec['height'])} m{centre}"
    return f"polygon with {len(spec['vertices'])} vertices"
