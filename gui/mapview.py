"""The map: matplotlib embedded in tkinter with the navigation toolbar.

All trees are one scatter collection.  Changing the filter, colours or
zoom updates its offsets, colours and sizes in place.  Layer shapes are
drawn as translucent patches on top.  Mouse events are forwarded to the
active painting tool (see ``gui.tools``); with no tool active a click
selects the nearest tree.
"""

import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")

import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from . import figures  # noqa: E402

PICK_TOLERANCE_PX = 10


class MapView:
    """Map widget: window outline, trees, layer overlays, selection and events."""

    def __init__(self, parent, on_tree_selected=None, on_view_changed=None, on_error=None):
        self.frame = ttk.Frame(parent)
        self.on_tree_selected = on_tree_selected
        self.on_view_changed = on_view_changed
        self.tool = None                      # object with press/motion/release/key methods

        self.figure = Figure(figsize=(7, 7), dpi=100)
        self.figure.set_facecolor("#fbfaf6")
        self.ax = self.figure.add_axes([0.08, 0.06, 0.9, 0.92])
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        bar = ttk.Frame(self.frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, bar, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.LEFT, fill=tk.X)
        ttk.Button(bar, text="Fit", width=5, command=self.fit).pack(side=tk.LEFT, padx=4)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.trees = None
        self.truth = None
        self.window = None
        self.visible = np.zeros(0, dtype=np.intp)   # indices of drawn trees
        self.colour_by = "species"
        self.size_by = "dbh"
        self.hide_coordinates = False
        self._kdtree = None
        self._diameters = np.zeros(0)
        self._window_patch = None
        self._overlays = []
        self._legend = None
        self.scatter = self.ax.scatter([], [], s=[], linewidths=0, zorder=2)
        (self._marker,) = self.ax.plot([], [], "o", ms=14, mfc="none", mec="#d62728", mew=2, zorder=5)
        self.ax.set_aspect("equal")
        self.ax.format_coord = self._format_coord
        self._limits_job = None

        self.ax.callbacks.connect("xlim_changed", self._on_limits)
        self.ax.callbacks.connect("ylim_changed", self._on_limits)
        self.canvas.mpl_connect("resize_event", lambda event: self._on_limits(self.ax))
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        if on_error is not None:
            # matplotlib would only print errors in event handlers; show them instead
            self.canvas.callbacks.exception_handler = on_error
            self.ax.callbacks.exception_handler = on_error

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def set_forest(self, trees, window, truth=None, fit=False):
        """Show a new tree table and window; keeps the zoom unless ``fit``."""
        first = self.window is None
        self.trees, self.truth, self.window = trees, truth, window
        if self._window_patch is not None:
            self._window_patch.remove()
        self._window_patch = figures.geometry_patch(
            window, facecolor="#f4f1e8", edgecolor="#555555", linewidth=1.0, zorder=0)
        if self._window_patch is not None:
            self.ax.add_patch(self._window_patch)
        self._diameters = figures.marker_diameters_m(trees, self.size_by)
        self.set_visible(np.arange(len(trees)), redraw=False)
        if first or fit:
            self.fit(redraw=False)
        self.refresh_style()

    def set_visible(self, index, redraw=True):
        """Draw only the trees at ``index`` (the filter result)."""
        self.visible = np.asarray(index, dtype=np.intp)
        xy = self._xy()[self.visible]
        self.scatter.set_offsets(xy if len(xy) else np.empty((0, 2)))
        self._kdtree = cKDTree(xy) if len(xy) else None
        self.select_tree(None, notify=False)
        if redraw:
            self.refresh_style()

    def _xy(self):
        if self.trees is None:
            return np.empty((0, 2))
        return np.column_stack([self.trees["X"].to_numpy(float), self.trees["Y"].to_numpy(float)])

    def refresh_style(self):
        """Recompute colours, sizes and legend of the drawn trees."""
        if self.trees is None:
            return
        rgba, legend = figures.tree_colours(self.trees, self.colour_by, self.truth)
        self.scatter.set_facecolors(rgba[self.visible] if len(self.visible) else np.empty((0, 4)))
        self._update_sizes()
        if self._legend is not None:
            self._legend.remove()
            self._legend = None
        handles = [Line2D([], [], ls="", marker="o", ms=7, mfc=c, mec="none",
                                           label=label) for label, c in legend[:12]]
        if handles:
            self._legend = self.ax.legend(handles=handles, loc="upper right", fontsize=8,
                                          framealpha=0.85)
        self.canvas.draw_idle()

    def set_colour_by(self, value):
        self.colour_by = value
        self.refresh_style()

    def set_size_by(self, value):
        self.size_by = value
        if self.trees is not None:
            self._diameters = figures.marker_diameters_m(self.trees, value)
        self.refresh_style()

    def set_hide_coordinates(self, hide):
        """Remove axis ticks and cursor coordinates (for confidential data)."""
        self.hide_coordinates = bool(hide)
        figures.set_hide_coordinates(self.ax, self.hide_coordinates)
        self.toolbar.set_message("")
        self.canvas.draw_idle()

    def _format_coord(self, x, y):
        return "" if self.hide_coordinates else f"x = {x:,.1f} m   y = {y:,.1f} m"

    # ------------------------------------------------------------------
    # View
    # ------------------------------------------------------------------

    def fit(self, redraw=True):
        """Zoom to the whole window."""
        if self.window is None:
            return
        minx, miny, maxx, maxy = self.window.bounds
        pad = 0.03 * max(maxx - minx, maxy - miny, 1.0)
        self.ax.set_xlim(minx - pad, maxx + pad)
        self.ax.set_ylim(miny - pad, maxy + pad)
        self.toolbar.update()          # make "home" the fitted view
        if redraw:
            self.canvas.draw_idle()

    def view_box(self):
        """Current axis limits as (minx, miny, maxx, maxy)."""
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def points_per_metre(self):
        return figures.points_per_metre(self.ax)

    def _update_sizes(self):
        if len(self.visible):
            self.scatter.set_sizes(figures.marker_sizes(self._diameters[self.visible],
                                                        self.points_per_metre()))
        for artist in list(self.ax.artists):
            if artist.get_gid() == "scalebar":
                artist.remove()
        figures.add_scale_bar(self.ax).set_gid("scalebar")

    def _on_limits(self, ax):
        # x and y limits change one after the other; update once both are set.
        if self._limits_job is None:
            self._limits_job = self.frame.after_idle(self._apply_limits)

    def _apply_limits(self):
        self._limits_job = None
        self._update_sizes()
        self.canvas.draw_idle()
        if self.on_view_changed:
            self.on_view_changed()

    def _on_scroll(self, event):
        """Mouse-wheel zoom around the cursor."""
        if event.inaxes is not self.ax or event.xdata is None:
            return
        factor = 1 / 1.25 if event.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        self.ax.set_xlim(cx + (x0 - cx) * factor, cx + (x1 - cx) * factor)
        self.ax.set_ylim(cy + (y0 - cy) * factor, cy + (y1 - cy) * factor)

    # ------------------------------------------------------------------
    # Layer overlays
    # ------------------------------------------------------------------

    def set_overlays(self, items):
        """Draw layer shapes: items are (geometry, colour, highlighted)."""
        for artist in self._overlays:
            artist.remove()
        self._overlays = []
        self._highlighted = []
        for geom, colour, highlighted in items:
            fill = figures.geometry_patch(geom, facecolor=colour, edgecolor="none",
                                          alpha=0.28 if highlighted else 0.12, zorder=1)
            edge = figures.geometry_patch(geom, facecolor="none", edgecolor=colour,
                                          linewidth=2.4 if highlighted else 1.0,
                                          linestyle="-" if highlighted else "--", zorder=4)
            for artist in (fill, edge):
                if artist is not None:
                    self.ax.add_patch(artist)
                    self._overlays.append(artist)
                    if highlighted:
                        self._highlighted.append(artist)
        self.canvas.draw_idle()

    def selected_overlay_artists(self):
        """Patches of the highlighted layer (moved while dragging)."""
        return list(getattr(self, "_highlighted", []))

    # ------------------------------------------------------------------
    # Selection and events
    # ------------------------------------------------------------------

    def navigating(self):
        """True while the toolbar's pan or zoom mode is active."""
        return bool(getattr(self.toolbar, "mode", ""))

    def select_tree(self, index, notify=True):
        """Mark tree ``index`` (row of the tree table) or clear the mark."""
        if index is None or self.trees is None:
            self._marker.set_data([], [])
        else:
            self._marker.set_data([self.trees["X"].iat[index]], [self.trees["Y"].iat[index]])
        self.canvas.draw_idle()
        if notify and self.on_tree_selected:
            self.on_tree_selected(index)

    def nearest_tree(self, x, y):
        """Row index of the drawn tree nearest to (x, y) within the pick tolerance."""
        if self._kdtree is None:
            return None
        tolerance_m = PICK_TOLERANCE_PX * 72.0 / self.figure.dpi / max(self.points_per_metre(), 1e-9)
        dist, pos = self._kdtree.query([x, y])
        if not np.isfinite(dist) or dist > max(tolerance_m, 0.5):
            return None
        return int(self.visible[pos])

    def _on_press(self, event):
        if event.inaxes is not self.ax or self.navigating():
            return
        if self.tool is not None and self.tool.press(event):
            return
        if event.button == 1 and event.xdata is not None:
            self.select_tree(self.nearest_tree(event.xdata, event.ydata))

    def _on_motion(self, event):
        if self.tool is not None and not self.navigating():
            self.tool.motion(event)

    def _on_release(self, event):
        if self.tool is not None and not self.navigating():
            self.tool.release(event)

    def _on_key(self, event):
        if self.tool is not None:
            self.tool.key(event)

    def redraw(self):
        self.canvas.draw_idle()

    def save_image(self, path, dpi=150):
        """Save the current map view (respects hidden coordinates)."""
        self._marker.set_visible(False)
        try:
            self.figure.savefig(path, dpi=dpi, facecolor=self.figure.get_facecolor())
        finally:
            self._marker.set_visible(True)
