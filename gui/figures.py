"""Matplotlib drawing shared by the GUI and the command line.

This module imports matplotlib but never tkinter, so the command line can
render PNG, SVG and PDF maps headless.
"""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

SPECIES_NAMES = {"1": "pine", "2": "spruce", "3": "broadleaved"}
SPECIES_COLOURS = {"1": "#c8962e", "2": "#1f5c3a", "3": "#7fbf4d"}
STATUS_COLOURS = {"Alive": "#2f7d4a", "Dead": "#8a4b2a"}
TYPE_COLOURS = {"ITD": "#2458a6", "Simulated": "#b4b4b4"}
OTHER_COLOUR = "#999999"
LAYER_COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                 "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
                 "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d")
COLOUR_BY = ("species", "status", "type", "layer")
SIZE_BY = ("fixed", "dbh", "crown")
MIN_MARKER_PT, MAX_MARKER_PT = 0.8, 60.0
MAX_MARKER_M = 5.0      # at overview zoom, no marker is drawn wider than 5 m


def _hex_to_rgba(colour, alpha=1.0):
    colour = colour.lstrip("#")
    return tuple(int(colour[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (alpha,)


def layer_colour(index):
    """Colour of the layer at ``index`` (cycles through a fixed palette)."""
    return LAYER_COLOURS[index % len(LAYER_COLOURS)]


def tree_colours(trees, colour_by="species", truth=None):
    """RGBA colours for every tree and the legend entries [(label, colour)]."""
    n = len(trees)
    if colour_by == "layer" and truth is not None and len(truth) == n:
        values = truth["layer_id"].astype(str).to_numpy()
        names = truth["layer_name"].astype(str).to_numpy()
        order = list(dict.fromkeys(values))
        mapping = {v: layer_colour(i) for i, v in enumerate(order)}
        labels = {v: names[np.argmax(values == v)] for v in order}
        legend = [(labels[v], mapping[v]) for v in order]
    else:
        column, mapping, labels = {
            "status": ("Status", STATUS_COLOURS, {}),
            "type": ("Type", TYPE_COLOURS, {}),
        }.get(colour_by, ("Species", SPECIES_COLOURS, SPECIES_NAMES))
        values = trees[column].astype(str).to_numpy()
        present = [v for v in mapping if (values == v).any()]
        legend = [(labels.get(v, v), mapping[v]) for v in present]
        if (~np.isin(values, list(mapping))).any():
            legend.append(("other", OTHER_COLOUR))
    rgba = np.tile(_hex_to_rgba(OTHER_COLOUR), (n, 1))
    for value, colour in mapping.items():
        rgba[values == value] = _hex_to_rgba(colour)
    return rgba, legend


def filter_mask(trees, dbh_min=None, dbh_max=None, species=None, status=None, types=None):
    """Boolean mask of trees passing the viewer filter.

    ``species``, ``status`` and ``types`` are sets of allowed values (None =
    all); species codes other than "1", "2", "3" count as "other".  Trees
    with unknown dbh pass the dbh range.
    """
    mask = np.ones(len(trees), dtype=bool)
    dbh = trees["DBH"].to_numpy(dtype=float)
    known = np.isfinite(dbh)
    if dbh_min is not None:
        mask &= ~known | (dbh >= dbh_min)
    if dbh_max is not None:
        mask &= ~known | (dbh <= dbh_max)
    if species is not None:
        codes = trees["Species"].astype(str).to_numpy()
        codes = np.where(np.isin(codes, list(SPECIES_NAMES)), codes, "other")
        mask &= np.isin(codes, list(species))
    if status is not None:
        mask &= np.isin(trees["Status"].astype(str).to_numpy(), list(status))
    if types is not None:
        mask &= np.isin(trees["Type"].astype(str).to_numpy(), list(types))
    return mask


def marker_diameters_m(trees, size_by="dbh"):
    """Marker diameter of every tree in metres (before zoom scaling)."""
    dbh = trees["DBH"].to_numpy(dtype=float)
    dbh = np.where(np.isfinite(dbh), dbh, 10.0)
    if size_by == "fixed":
        return np.full(len(trees), 1.5)
    if size_by == "crown":
        crown = trees["Latvus_d"].to_numpy(dtype=float)
        return np.where(np.isfinite(crown), crown, 0.8)
    return 0.12 * dbh


def marker_sizes(diameters_m, points_per_metre):
    """Scatter sizes (points^2) for diameters in metres at the current zoom.

    Markers keep their true size in metres when zoomed in; at overview zoom
    they are kept at least ``MIN_MARKER_PT`` wide and at most a few points,
    so single big trees do not hide the pattern.
    """
    upper = min(MAX_MARKER_PT, max(4.0, MAX_MARKER_M * points_per_metre))
    pt = np.clip(diameters_m * points_per_metre, MIN_MARKER_PT, upper)
    return pt ** 2


def points_per_metre(ax):
    """How many typographic points one metre spans in the axes' current view."""
    ax.apply_aspect()           # equal aspect shrinks the axes box; measure the real one
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    dpi = ax.figure.dpi
    width_pt = bbox.width * 72.0 / dpi
    return width_pt / max(abs(x1 - x0), 1e-9)


def geometry_path(geom):
    """A matplotlib Path for a (multi)polygon, holes included."""
    vertices, codes = [], []
    for part in shapely.get_parts(geom):
        if not isinstance(part, Polygon) or part.is_empty:
            continue
        part = orient(part, 1.0)
        for ring in [part.exterior, *part.interiors]:
            xy = np.asarray(ring.coords)
            vertices.append(xy)
            codes.append(np.r_[Path.MOVETO, np.full(len(xy) - 2, Path.LINETO), Path.CLOSEPOLY])
    if not vertices:
        return None
    return Path(np.vstack(vertices), np.concatenate(codes).astype(Path.code_type))


def geometry_patch(geom, **kwargs):
    """A PathPatch for a polygonal geometry, or None if empty."""
    path = geometry_path(geom)
    return PathPatch(path, **kwargs) if path is not None else None


def set_hide_coordinates(ax, hide):
    """Remove (or restore) ticks and tick labels; a scale bar stays visible."""
    if hide:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.xaxis.set_major_locator(_auto_locator())
        ax.yaxis.set_major_locator(_auto_locator())
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")


def _auto_locator():
    from matplotlib.ticker import AutoLocator
    return AutoLocator()


def nice_length(span):
    """A round scale-bar length of roughly a fifth of ``span`` metres."""
    target = span / 5.0
    if target <= 0:
        return 1.0
    power = 10 ** np.floor(np.log10(target))
    for step in (1, 2, 5, 10):
        if step * power >= target:
            return float(step * power)
    return float(10 * power)


def add_scale_bar(ax):
    """Add a scale bar sized to the current x-limits; returns the artist."""
    x0, x1 = ax.get_xlim()
    length = nice_length(abs(x1 - x0))
    label = f"{length:g} m"
    bar = AnchoredSizeBar(ax.transData, length, label, "lower right", pad=0.4,
                          borderpad=0.6, sep=3, frameon=True, size_vertical=0)
    bar.patch.set_alpha(0.75)
    ax.add_artist(bar)
    return bar


def draw_map(ax, trees, window, colour_by="species", size_by="dbh", truth=None,
             hide_coordinates=False, title=None, overlays=()):
    """Draw window, optional layer overlays and trees on ``ax``; returns the scatter."""
    ax.set_aspect("equal")
    patch = geometry_patch(window, facecolor="#f4f1e8", edgecolor="#555555", linewidth=1.0, zorder=0)
    if patch is not None:
        ax.add_patch(patch)
    for geom, colour in overlays:
        fill = geometry_patch(geom, facecolor=colour, alpha=0.12, edgecolor="none", zorder=1)
        edge = geometry_patch(geom, facecolor="none", edgecolor=colour, linewidth=1.2, zorder=3)
        for artist in (fill, edge):
            if artist is not None:
                ax.add_patch(artist)
    minx, miny, maxx, maxy = window.bounds
    pad = 0.03 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    rgba, legend = tree_colours(trees, colour_by, truth)
    sizes = marker_sizes(marker_diameters_m(trees, size_by), points_per_metre(ax))
    scatter = ax.scatter(trees["X"].to_numpy(float), trees["Y"].to_numpy(float), s=sizes,
                         c=rgba, linewidths=0, zorder=2)
    for label, colour in legend[:12]:
        ax.scatter([], [], s=30, c=[colour], label=label)
    if legend:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85, markerscale=1.0)
    set_hide_coordinates(ax, hide_coordinates)
    add_scale_bar(ax)
    if title:
        ax.set_title(title, fontsize=10)
    return scatter


def save_map(path, trees, window, colour_by="species", size_by="dbh", truth=None,
             hide_coordinates=False, title=None, overlays=(), size_in=8.0, dpi=150):
    """Render the map to a PNG, SVG or PDF file (format from the extension)."""
    fig = Figure(figsize=(size_in, size_in), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.08, 0.06, 0.88, 0.88])
    draw_map(ax, trees, window, colour_by, size_by, truth, hide_coordinates, title, overlays)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    return path
