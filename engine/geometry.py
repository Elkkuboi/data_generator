"""Windows and layer shapes as shapely geometries.

A window is the whole area; every layer shape is clipped to it.  Shapes are
plain recipe dicts:

    {"kind": "window"}
    {"kind": "circle", "center": [x, y], "radius": r}
    {"kind": "rectangle", "center": [x, y], "width": w, "height": h}
    {"kind": "polygon", "vertices": [[x, y], ...]}
    {"kind": "corridor", "points": [[x, y], ...], "width": w}

A corridor is a polyline buffered to its width with round caps; freehand
brush strokes, trails and roads are all corridors.  Point-in-shape tests use
shapely's vectorised ``contains_xy`` / ``intersects_xy`` on prepared
geometries.
"""

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

CIRCLE_QUAD_SEGS = 64      # segments per quarter circle (window circles)
SHAPE_QUAD_SEGS = 16       # segments per quarter circle (layer shapes, corridor caps)
WINDOW_KINDS = ("circle", "rectangle", "polygon")
SHAPE_KINDS = ("window", "circle", "rectangle", "polygon", "corridor")


def _polygonal(geom):
    """Keep only the polygon parts of a geometry (drops stray lines and points)."""
    if geom.is_empty:
        return Polygon()
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    parts = [g for g in shapely.get_parts(geom) if isinstance(g, (Polygon, MultiPolygon))]
    if not parts:
        return Polygon()
    return shapely.union_all(parts)


def _polygon(vertices):
    poly = Polygon(vertices)
    if not poly.is_valid:
        poly = _polygonal(shapely.make_valid(poly))
    return poly


def _rectangle(center, width, height):
    cx, cy = center
    return box(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)


def corridor(points, width, quad_segs=SHAPE_QUAD_SEGS):
    """Polyline buffered to ``width`` with round caps (a single point gives a disc)."""
    pts = [tuple(p) for p in points]
    line = Point(pts[0]) if len(pts) == 1 else LineString(pts)
    return line.buffer(width / 2.0, quad_segs=quad_segs)


def raw_shape(spec):
    """Shapely geometry of a shape spec, not clipped (``window`` is not allowed)."""
    kind = spec["kind"]
    if kind == "circle":
        return Point(spec["center"]).buffer(spec["radius"], quad_segs=SHAPE_QUAD_SEGS)
    if kind == "rectangle":
        return _rectangle(spec["center"], spec["width"], spec["height"])
    if kind == "polygon":
        return _polygon(spec["vertices"])
    if kind == "corridor":
        return corridor(spec["points"], spec["width"])
    raise ValueError(f"unknown shape kind {kind!r}")


def window_geometry(spec):
    """Prepared shapely polygon of a window spec."""
    kind = spec["kind"]
    if kind == "circle":
        geom = Point(spec["center"]).buffer(spec["radius"], quad_segs=CIRCLE_QUAD_SEGS)
    elif kind in ("rectangle", "polygon"):
        geom = raw_shape(spec)
    else:
        raise ValueError(f"unknown window kind {kind!r}")
    shapely.prepare(geom)
    return geom


def shape_geometry(spec, window):
    """Prepared geometry of a layer shape clipped to the window (may be empty)."""
    if spec["kind"] == "window":
        return window
    with np.errstate(invalid="ignore"):     # old GEOS warns on disjoint inputs
        geom = _polygonal(raw_shape(spec).intersection(window))
    shapely.prepare(geom)
    return geom


def points_inside(geom, x, y):
    """Boolean mask of points strictly inside ``geom``."""
    if geom.is_empty or len(x) == 0:
        return np.zeros(len(x), dtype=bool)
    return shapely.contains_xy(geom, x, y)


def points_touching(geom, x, y):
    """Boolean mask of points inside or on the boundary of ``geom``."""
    if geom.is_empty or len(x) == 0:
        return np.zeros(len(x), dtype=bool)
    return shapely.intersects_xy(geom, x, y)


def area_ha(geom):
    """Area of a geometry in hectares."""
    return geom.area / 10_000.0


def translate_shape(spec, dx, dy):
    """Return a copy of a shape spec moved by (dx, dy)."""
    out = dict(spec)
    if "center" in spec:
        out["center"] = [spec["center"][0] + dx, spec["center"][1] + dy]
    for key in ("vertices", "points"):
        if key in spec:
            out[key] = [[p[0] + dx, p[1] + dy] for p in spec[key]]
    return out


def snap_inside(x, y, step, offset, geom):
    """Snap points to a lattice (``offset + k * step``) keeping them inside ``geom``.

    Each point moves to the nearest of the four lattice corners around it
    that lies strictly inside the geometry.  Coordinates are rounded to two
    decimals (``offset`` and ``step`` must be multiples of 0.01).  Returns the
    new x, y and a mask of points that could be placed; points whose four
    corners all fall outside are dropped by the caller.
    """
    n = len(x)
    if n == 0:
        return x, y, np.zeros(0, dtype=bool)
    ox, oy = offset
    fx = np.floor((x - ox) / step)
    fy = np.floor((y - oy) / step)
    cx = np.round(ox + step * (fx[:, None] + np.array([0, 1, 0, 1])), 2)
    cy = np.round(oy + step * (fy[:, None] + np.array([0, 0, 1, 1])), 2)
    inside = shapely.contains_xy(geom, cx.ravel(), cy.ravel()).reshape(n, 4)
    dist = (cx - x[:, None]) ** 2 + (cy - y[:, None]) ** 2
    dist[~inside] = np.inf
    best = np.argmin(dist, axis=1)
    rows = np.arange(n)
    ok = np.isfinite(dist[rows, best])
    return cx[rows, best], cy[rows, best], ok


def shape_outline_xy(geom):
    """List of (x, y) coordinate arrays for every exterior and interior ring."""
    rings = []
    for part in shapely.get_parts(geom):
        if isinstance(part, Polygon) and not part.is_empty:
            rings.append(np.asarray(part.exterior.coords))
            rings.extend(np.asarray(r.coords) for r in part.interiors)
    return rings
