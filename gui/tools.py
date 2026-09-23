"""Painting tools: turn mouse gestures on the map into layer shapes.

Every tool receives matplotlib mouse events from ``MapView`` and, when a
shape is finished, calls ``on_shape(spec)`` with a recipe shape dict.  The
shapes are:

* Brush      - freehand stroke, stored as a corridor (path + brush diameter)
* Circle     - press at the centre, drag to the radius
* Rectangle  - drag from corner to corner
* Polygon    - click vertices, double-click (or Enter) to close
* Corridor   - click points along a line, double-click (or Enter) to finish

Escape cancels the shape being drawn; Backspace removes the last point.
``SelectTool`` selects trees or layers by clicking and moves the selected
layer's shape by dragging it.
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
from shapely import contains_xy
from shapely.geometry import LineString

MIN_SIZE_M = 0.5


def _round(value):
    return round(float(value), 2)


def _pt(x, y):
    return [_round(x), _round(y)]


class Tool:
    """Base class: no-op handlers; ``press`` returns True when it used the event."""

    name = "tool"

    def __init__(self, mapview, on_shape, colour="#d62728"):
        self.map = mapview
        self.ax = mapview.ax
        self.on_shape = on_shape
        self.colour = colour
        self.artists = []

    def activate(self):
        pass

    def deactivate(self):
        self.cancel()

    def press(self, event):
        return False

    def motion(self, event):
        pass

    def release(self, event):
        pass

    def key(self, event):
        if event.key == "escape":
            self.cancel()

    def cancel(self):
        for artist in self.artists:
            artist.remove()
        self.artists = []
        self.map.redraw()

    def _add(self, artist):
        self.artists.append(artist)
        return artist

    def finish(self, spec):
        self.cancel()
        if spec is not None:
            self.on_shape(spec)


class BrushTool(Tool):
    """Freehand strokes with a round brush of adjustable diameter."""

    name = "brush"

    def __init__(self, mapview, on_shape, get_diameter, colour="#d62728"):
        super().__init__(mapview, on_shape, colour)
        self.get_diameter = get_diameter
        self.points = None
        self.line = None

    def _linewidth(self):
        return max(self.get_diameter() * self.map.points_per_metre(), 1.0)

    def press(self, event):
        if event.button != 1 or event.xdata is None:
            return False
        self.cancel()
        self.points = [(event.xdata, event.ydata)]
        self.line = self._add(Line2D([event.xdata], [event.ydata], lw=self._linewidth(),
                                     color=self.colour, alpha=0.45, solid_capstyle="round",
                                     solid_joinstyle="round", marker="o",
                                     ms=self._linewidth(), mec="none", zorder=6))
        self.ax.add_line(self.line)
        self.map.redraw()
        return True

    def motion(self, event):
        if self.points is None or event.xdata is None:
            return
        last = self.points[-1]
        step = max(0.2 * self.get_diameter(), 0.3)
        if np.hypot(event.xdata - last[0], event.ydata - last[1]) >= step:
            self.points.append((event.xdata, event.ydata))
            xs, ys = zip(*self.points)
            self.line.set_data(xs, ys)
            self.line.set_marker("None")
            self.map.redraw()

    def release(self, event):
        if self.points is None:
            return
        points, self.points = self.points, None
        diameter = float(self.get_diameter())
        if len(points) > 2:
            simple = LineString(points).simplify(0.05 * diameter)
            points = list(simple.coords)
        self.finish({"kind": "corridor", "points": [_pt(x, y) for x, y in points],
                     "width": _round(diameter)})

    def cancel(self):
        self.points = None
        super().cancel()


class CircleTool(Tool):
    name = "circle"

    def __init__(self, mapview, on_shape, colour="#d62728"):
        super().__init__(mapview, on_shape, colour)
        self.centre = None
        self.patch = None

    def press(self, event):
        if event.button != 1 or event.xdata is None:
            return False
        self.cancel()
        self.centre = (event.xdata, event.ydata)
        self.patch = self._add(Circle(self.centre, 0.0, facecolor=self.colour, alpha=0.3,
                                      edgecolor=self.colour, zorder=6))
        self.ax.add_patch(self.patch)
        return True

    def motion(self, event):
        if self.centre is None or event.xdata is None:
            return
        self.patch.set_radius(np.hypot(event.xdata - self.centre[0], event.ydata - self.centre[1]))
        self.map.redraw()

    def release(self, event):
        if self.centre is None:
            return
        radius = self.patch.get_radius()
        centre, self.centre = self.centre, None
        self.finish({"kind": "circle", "center": _pt(*centre), "radius": _round(radius)}
                    if radius >= MIN_SIZE_M else None)

    def cancel(self):
        self.centre = None
        super().cancel()


class RectangleTool(Tool):
    name = "rectangle"

    def __init__(self, mapview, on_shape, colour="#d62728"):
        super().__init__(mapview, on_shape, colour)
        self.start = None
        self.patch = None

    def press(self, event):
        if event.button != 1 or event.xdata is None:
            return False
        self.cancel()
        self.start = (event.xdata, event.ydata)
        self.patch = self._add(Rectangle(self.start, 0.0, 0.0, facecolor=self.colour, alpha=0.3,
                                         edgecolor=self.colour, zorder=6))
        self.ax.add_patch(self.patch)
        return True

    def motion(self, event):
        if self.start is None or event.xdata is None:
            return
        x0, y0 = self.start
        self.patch.set_bounds(min(x0, event.xdata), min(y0, event.ydata),
                              abs(event.xdata - x0), abs(event.ydata - y0))
        self.map.redraw()

    def release(self, event):
        if self.start is None:
            return
        self.start = None
        x, y = self.patch.get_xy()
        w, h = self.patch.get_width(), self.patch.get_height()
        ok = w >= MIN_SIZE_M and h >= MIN_SIZE_M
        self.finish({"kind": "rectangle", "center": _pt(x + w / 2, y + h / 2),
                     "width": _round(w), "height": _round(h)} if ok else None)

    def cancel(self):
        self.start = None
        super().cancel()


class _ClickPathTool(Tool):
    """Shared logic of polygon and corridor tools: click points, double-click to finish."""

    minimum = 2

    def __init__(self, mapview, on_shape, colour="#d62728"):
        super().__init__(mapview, on_shape, colour)
        self.points = []
        self.line = None
        self.rubber = None

    def _style(self):
        return {"color": self.colour, "lw": 1.5, "marker": "o", "ms": 4}

    def press(self, event):
        if event.xdata is None:
            return False
        if event.button == 3:
            self.complete()
            return True
        if event.button != 1:
            return False
        if event.dblclick:
            self.complete()
            return True
        if self.line is None:
            self.line = self._add(Line2D([], [], zorder=6, **self._style()))
            self.rubber = self._add(Line2D([], [], color=self.colour, lw=1.0, ls="--", zorder=6))
            self.ax.add_line(self.line)
            self.ax.add_line(self.rubber)
        self.points.append((event.xdata, event.ydata))
        self._update_line()
        return True

    def _update_line(self):
        if self.line is not None:
            xs, ys = zip(*self.points) if self.points else ([], [])
            self.line.set_data(xs, ys)
        self.map.redraw()

    def motion(self, event):
        if self.points and event.xdata is not None and self.rubber is not None:
            last = self.points[-1]
            self.rubber.set_data([last[0], event.xdata], [last[1], event.ydata])
            self.map.redraw()

    def key(self, event):
        if event.key == "enter":
            self.complete()
        elif event.key == "backspace" and self.points:
            self.points.pop()
            self._update_line()
        else:
            super().key(event)

    def complete(self):
        points = self._distinct()
        if len(points) < self.minimum:
            self.cancel()
            return
        self.finish(self.make_spec(points))

    def _distinct(self):
        out = []
        for p in self.points:
            if not out or np.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-6:
                out.append(p)
        return out

    def make_spec(self, points):
        raise NotImplementedError

    def cancel(self):
        self.points = []
        self.line = self.rubber = None
        super().cancel()


class PolygonTool(_ClickPathTool):
    name = "polygon"
    minimum = 3

    def make_spec(self, points):
        return {"kind": "polygon", "vertices": [_pt(x, y) for x, y in points]}


class CorridorTool(_ClickPathTool):
    name = "corridor"
    minimum = 2

    def __init__(self, mapview, on_shape, get_width, colour="#d62728"):
        super().__init__(mapview, on_shape, colour)
        self.get_width = get_width

    def _style(self):
        lw = max(self.get_width() * self.map.points_per_metre(), 1.5)
        return {"color": self.colour, "lw": lw, "alpha": 0.45, "solid_capstyle": "round",
                "solid_joinstyle": "round"}

    def make_spec(self, points):
        return {"kind": "corridor", "points": [_pt(x, y) for x, y in points],
                "width": _round(self.get_width())}


class SelectTool(Tool):
    """Click: select the tree and the topmost layer under the cursor.
    Drag inside the selected layer's shape: move the shape."""

    name = "select"

    def __init__(self, mapview, pick_layer, selected_geometry, on_move):
        super().__init__(mapview, on_shape=None)
        self.pick_layer = pick_layer              # (x, y) -> layer index or None
        self.selected_geometry = selected_geometry   # () -> shapely geometry or None
        self.on_move = on_move                    # (dx, dy) -> None
        self.start = None
        self.moved = False

    def press(self, event):
        if event.button != 1 or event.xdata is None:
            return False
        geom = self.selected_geometry()
        if geom is not None and not geom.is_empty and contains_xy(geom, event.xdata, event.ydata):
            self.start = (event.xdata, event.ydata)
            self.moved = False
            return False          # still let the map select a tree on a simple click
        self.pick_layer(event.xdata, event.ydata)
        return False

    def motion(self, event):
        if self.start is None or event.xdata is None:
            return
        dx, dy = event.xdata - self.start[0], event.ydata - self.start[1]
        if not self.moved and np.hypot(dx, dy) * self.map.points_per_metre() < 4:
            return
        self.moved = True
        shift = Affine2D().translate(dx, dy) + self.ax.transData
        for artist in self.map.selected_overlay_artists():
            artist.set_transform(shift)
        self.map.redraw()

    def release(self, event):
        if self.start is None:
            return
        start, self.start = self.start, None
        if self.moved and event.xdata is not None:
            self.on_move(event.xdata - start[0], event.ydata - start[1])
        elif self.moved:
            for artist in self.map.selected_overlay_artists():
                artist.set_transform(self.ax.transData)
            self.map.redraw()
        self.moved = False

    def cancel(self):
        self.start = None
        self.moved = False
