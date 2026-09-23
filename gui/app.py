"""Main window: viewer and painting editor (tkinter).

Layout: tools and view controls on the left, the map in the centre, layers
and statistics on the right, main buttons and a status bar at the bottom.
The window has two modes:

* editor - shows the forest generated from the current recipe;
* viewer - shows a CSV or RDS file (synthetic or real); coordinates are
  hidden by default because real inventory coordinates are confidential.

Errors never show a traceback: they become dialogs, are written to a log
file, and the program keeps running.
"""

import os
import sys
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace

import numpy as np
import shapely
from shapely.geometry import box

from engine.generate import generate
from engine.io import TreeFileError, read_trees, read_truth, sidecar_paths, window_for_trees
from engine.recipe import RecipeError, new_recipe
from engine.report import window_description

from . import figures
from .mapview import MapView
from .stats_panel import StatsPanel

APP_NAME = "synthforest"
FILTER_DELAY_MS = 250
VIEW_STATS_DELAY_MS = 500


def config_dir():
    """Folder for the recovery file and the error log."""
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, APP_NAME)
    return os.path.join(os.path.expanduser("~"), "." + APP_NAME)


def log_error(text):
    """Append a traceback to the error log; never raises."""
    try:
        os.makedirs(config_dir(), exist_ok=True)
        with open(os.path.join(config_dir(), "errors.log"), "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    except OSError:
        pass


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable frame; put children into ``.inner``."""

    def __init__(self, parent, width=250):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, width=width, highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._item = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._item, width=e.width))
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.inner.bind("<Enter>", lambda e: self._bind_wheel(True))
        self.inner.bind("<Leave>", lambda e: self._bind_wheel(False))

    def _bind_wheel(self, on):
        if on:
            self.canvas.bind_all("<MouseWheel>", self._on_wheel)
            self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-2, "units"))
            self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(2, "units"))
        else:
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.canvas.unbind_all(seq)

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120) or (-1 if event.delta > 0 else 1), "units")


class App:
    """The synthforest main window."""

    def __init__(self, root, view_file=None, recipe_file=None):
        self.root = root
        self.mode = "editor"
        self.recipe = new_recipe()
        self.recipe_path = None
        self.shown = None                 # SimpleNamespace(trees, truth, window, note, source)
        self.selected_tree = None
        self._filter_job = None
        self._view_job = None

        root.title(APP_NAME)
        root.geometry("1400x900")
        root.minsize(1000, 650)
        root.report_callback_exception = self._on_tk_error
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self._build()
        self._bind_keys()

        if view_file:
            root.after(50, lambda: self.open_tree_file(view_file))
        elif recipe_file:
            root.after(50, lambda: self.open_recipe(recipe_file))
        else:
            root.after(50, self.regenerate)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self):
        style = ttk.Style(self.root)
        if not sys.platform.startswith("win") and "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Status.TLabel", padding=(6, 2))
        style.configure("Head.TLabel", font=("TkDefaultFont", 10, "bold"))

        bottom = ttk.Frame(self.root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(bottom, textvariable=self.status, style="Status.TLabel", anchor="w",
                  relief="sunken").pack(side=tk.BOTTOM, fill=tk.X)
        self.buttons = ttk.Frame(bottom, padding=(4, 4))
        self.buttons.pack(side=tk.BOTTOM, fill=tk.X)
        self._build_buttons(self.buttons)

        panes = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)
        self.left = ScrollableFrame(panes, width=250)
        self.map = MapView(panes, on_tree_selected=self._on_tree_selected,
                           on_view_changed=self._on_view_changed)
        self.right = ttk.Notebook(panes, width=430)
        panes.add(self.left, weight=0)
        panes.add(self.map.frame, weight=1)
        panes.add(self.right, weight=0)

        self._build_left(self.left.inner)
        self._build_right(self.right)

    def _build_buttons(self, bar):
        self.main_buttons = {}

        def button(key, text, command, **pack):
            b = ttk.Button(bar, text=text, command=command)
            b.pack(side=tk.LEFT, padx=2, **pack)
            self.main_buttons[key] = b
            return b

        button("new", "New", self.new_recipe)
        button("open_recipe", "Open recipe…", self.open_recipe)
        button("save_recipe", "Save recipe…", self.save_recipe)
        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(bar, text="Seed").pack(side=tk.LEFT, padx=(2, 2))
        self.seed_var = tk.StringVar(value=str(self.recipe["seed"]))
        self.seed_entry = ttk.Entry(bar, textvariable=self.seed_var, width=10)
        self.seed_entry.pack(side=tk.LEFT)
        button("new_seed", "New seed", self.new_seed)
        button("generate", "Generate", self.regenerate)
        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=6)
        button("export", "Export…", self.export_dialog)
        button("open_file", "Open file…", self.open_tree_file)

    def _section(self, parent, title):
        ttk.Label(parent, text=title, style="Head.TLabel").pack(anchor="w", pady=(10, 2), padx=4)
        frame = ttk.Frame(parent, padding=(6, 0))
        frame.pack(fill=tk.X)
        return frame

    def _build_left(self, parent):
        self.editor_left = ttk.Frame(parent)
        self.editor_left.pack(fill=tk.X)
        self.viewer_left = ttk.Frame(parent)
        self.viewer_left.pack(fill=tk.X)
        self._build_view_controls(self.viewer_left)

    def _build_view_controls(self, parent):
        view = self._section(parent, "View")
        ttk.Label(view, text="Colour by").grid(row=0, column=0, sticky="w")
        self.colour_var = tk.StringVar(value="species")
        colour = ttk.Combobox(view, textvariable=self.colour_var, values=figures.COLOUR_BY,
                              state="readonly", width=12)
        colour.grid(row=0, column=1, sticky="ew", pady=1)
        colour.bind("<<ComboboxSelected>>", lambda e: self._on_colour())
        ttk.Label(view, text="Size by").grid(row=1, column=0, sticky="w")
        self.size_var = tk.StringVar(value="dbh")
        size = ttk.Combobox(view, textvariable=self.size_var, values=figures.SIZE_BY,
                            state="readonly", width=12)
        size.grid(row=1, column=1, sticky="ew", pady=1)
        size.bind("<<ComboboxSelected>>", lambda e: self.map.set_size_by(self.size_var.get()))
        self.hide_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(view, text="Hide coordinates", variable=self.hide_var,
                        command=self._on_hide).grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
        self.view_stats_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(view, text="Statistics for current view only", variable=self.view_stats_var,
                        command=self.update_stats).grid(row=3, column=0, columnspan=2, sticky="w")
        view.columnconfigure(1, weight=1)

        filt = self._section(parent, "Filter")
        ttk.Label(filt, text="DBH (cm)").grid(row=0, column=0, sticky="w")
        self.dbh_min_var, self.dbh_max_var = tk.StringVar(), tk.StringVar()
        ttk.Entry(filt, textvariable=self.dbh_min_var, width=6).grid(row=0, column=1)
        ttk.Label(filt, text="–").grid(row=0, column=2)
        ttk.Entry(filt, textvariable=self.dbh_max_var, width=6).grid(row=0, column=3)
        self.filter_vars = {}
        groups = (("Species", [("1", "pine"), ("2", "spruce"), ("3", "broadleaved"), ("other", "other")]),
                  ("Status", [("Alive", "alive"), ("Dead", "dead")]),
                  ("Type", [("ITD", "ITD"), ("Simulated", "simulated")]))
        row = 1
        for group, items in groups:
            ttk.Label(filt, text=group).grid(row=row, column=0, sticky="nw", pady=(4, 0))
            box_ = ttk.Frame(filt)
            box_.grid(row=row, column=1, columnspan=3, sticky="w", pady=(4, 0))
            for i, (value, label) in enumerate(items):
                var = tk.BooleanVar(value=True)
                self.filter_vars[group, value] = var
                ttk.Checkbutton(box_, text=label, variable=var, command=self.schedule_filter).grid(
                    row=i // 2, column=i % 2, sticky="w")
            row += 1
        for var in (self.dbh_min_var, self.dbh_max_var):
            var.trace_add("write", lambda *a: self.schedule_filter())
        ttk.Button(filt, text="Reset filter", command=self.reset_filter).grid(
            row=row, column=0, columnspan=4, sticky="w", pady=4)

        tree = self._section(parent, "Selected tree")
        self.tree_text = tk.Text(tree, height=14, width=30, font=("TkFixedFont", 9),
                                 borderwidth=1, relief="solid")
        self.tree_text.pack(fill=tk.X)
        self._show_tree_info(None)

    def _build_right(self, notebook):
        self.stats = StatsPanel(notebook)

    def _bind_keys(self):
        self.root.bind_all("<Control-o>", lambda e: self.open_tree_file())
        self.root.bind_all("<F5>", lambda e: self.regenerate())

    # ------------------------------------------------------------------
    # Errors and status
    # ------------------------------------------------------------------

    def _on_tk_error(self, exc_type, exc, tb):
        log_error("".join(traceback.format_exception(exc_type, exc, tb)))
        self.error("Something went wrong", f"{exc_type.__name__}: {exc}\n\nYour work is kept; "
                   f"details were written to {os.path.join(config_dir(), 'errors.log')}.")

    def error(self, title, message):
        messagebox.showerror(title, message, parent=self.root)

    def set_status(self, text):
        self.status.set(text)

    # ------------------------------------------------------------------
    # Showing data
    # ------------------------------------------------------------------

    def show(self, trees, window, truth=None, note="", source="", fit=False):
        """Display a tree table (generated or read from a file)."""
        self.shown = SimpleNamespace(trees=trees, truth=truth, window=window, note=note,
                                     source=source)
        self.map.set_forest(trees, window, truth, fit=fit)
        if self.colour_var.get() == "layer" and truth is None:
            self.colour_var.set("species")
        self.map.colour_by = self.colour_var.get()
        self.apply_filter()

    def _set_mode(self, mode):
        self.mode = mode
        editing = mode == "editor"
        for key in ("new_seed", "generate"):
            self.main_buttons[key].configure(state="normal" if editing else "disabled")
        self.seed_entry.configure(state="normal" if editing else "disabled")
        self.update_title()

    def update_title(self):
        if self.mode == "viewer" and self.shown is not None:
            self.root.title(f"{APP_NAME} – viewer – {os.path.basename(self.shown.source)}")
        else:
            name = self.recipe.get("name", "forest")
            where = f" ({os.path.basename(self.recipe_path)})" if self.recipe_path else ""
            self.root.title(f"{APP_NAME} – {name}{where}")

    # ------------------------------------------------------------------
    # Filter, view and statistics
    # ------------------------------------------------------------------

    def _filter_values(self):
        def number(var):
            text = var.get().strip().replace(",", ".")
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None
        allowed = {}
        for (group, value), var in self.filter_vars.items():
            allowed.setdefault(group, set())
            if var.get():
                allowed[group].add(value)
        return dict(dbh_min=number(self.dbh_min_var), dbh_max=number(self.dbh_max_var),
                    species=allowed["Species"], status=allowed["Status"], types=allowed["Type"])

    def schedule_filter(self):
        if self._filter_job is not None:
            self.root.after_cancel(self._filter_job)
        self._filter_job = self.root.after(FILTER_DELAY_MS, self.apply_filter)

    def reset_filter(self):
        self.dbh_min_var.set("")
        self.dbh_max_var.set("")
        for var in self.filter_vars.values():
            var.set(True)
        self.apply_filter()

    def visible_mask(self):
        if self.shown is None:
            return np.zeros(0, dtype=bool)
        values = self._filter_values()
        trees = self.shown.trees
        if not {"Alive", "Dead"} & set(trees["Status"]):      # a file without Status
            values["status"] = None
        if not {"ITD", "Simulated"} & set(trees["Type"]):
            values["types"] = None
        return figures.filter_mask(trees, **values)

    def apply_filter(self):
        self._filter_job = None
        if self.shown is None:
            return
        mask = self.visible_mask()
        self.map.set_visible(np.flatnonzero(mask))
        self._show_tree_info(None)
        self.update_stats()
        self.update_status()

    def _on_view_changed(self):
        if self.view_stats_var.get():
            if self._view_job is not None:
                self.root.after_cancel(self._view_job)
            self._view_job = self.root.after(VIEW_STATS_DELAY_MS, self.update_stats)

    def stats_region(self):
        """Region the statistics refer to: window, or window within the current view."""
        window = self.shown.window
        if self.view_stats_var.get():
            region = window.intersection(box(*self.map.view_box()))
            shapely.prepare(region)
            return region, "current view"
        return window, "whole area"

    def update_stats(self):
        self._view_job = None
        if self.shown is None:
            self.stats.clear()
            return
        trees = self.shown.trees.iloc[self.map.visible]
        region, scope = self.stats_region()
        if self.view_stats_var.get():
            inside = shapely.contains_xy(region, trees["X"].to_numpy(float), trees["Y"].to_numpy(float))
            trees = trees[inside]
        shown = len(trees)
        text = f"Scope: {scope}; {shown:,} of {len(self.shown.trees):,} trees pass the filter."
        self.stats.update(trees.reset_index(drop=True), region, self.shown.note, text)

    def update_status(self):
        if self.shown is None:
            self.set_status("No trees.")
            return
        n, shown = len(self.shown.trees), len(self.map.visible)
        density = n / (self.shown.window.area / 1e4) if self.shown.window.area > 0 else float("nan")
        part = f" ({shown:,} shown)" if shown != n else ""
        prefix = os.path.basename(self.shown.source) if self.mode == "viewer" else self.recipe["name"]
        extra = getattr(self, "status_extra", "")
        self.set_status(f"{prefix}: {n:,} trees{part} · {density:.0f} /ha · {self.shown.note}{extra}")

    def _on_colour(self):
        value = self.colour_var.get()
        if value == "layer" and (self.shown is None or self.shown.truth is None):
            self.error("Colour by layer", "Layer colours need the ground truth of a synthetic "
                       "forest (the <name>-truth.csv file next to the tree file).")
            self.colour_var.set(self.map.colour_by)
            return
        self.map.set_colour_by(value)

    def _on_hide(self):
        self.map.set_hide_coordinates(self.hide_var.get())
        self._show_tree_info(self.selected_tree)

    def _on_tree_selected(self, index):
        self._show_tree_info(index)

    def _show_tree_info(self, index):
        self.selected_tree = index
        text = self.tree_text
        text.configure(state="normal")
        text.delete("1.0", tk.END)
        if index is None or self.shown is None or index >= len(self.shown.trees):
            text.insert("1.0", "Click a tree on the map\n(with no painting tool\nand no pan/zoom).")
        else:
            row = self.shown.trees.iloc[index]
            lines = [f"{'row':<9}{index + 1}"]
            for name in row.index:
                value = row[name]
                if name in ("X", "Y") and self.hide_var.get():
                    shown = "hidden"
                elif isinstance(value, float):
                    shown = "NA" if np.isnan(value) else (f"{value:.4f}" if name == "V" else f"{value:.2f}")
                else:
                    shown = str(value)
                if name == "Species":
                    shown += f" ({figures.SPECIES_NAMES.get(shown, 'other')})"
                lines.append(f"{name:<9}{shown}")
            if self.shown.truth is not None:
                t = self.shown.truth.iloc[index]
                lines += ["", f"{'layer':<9}{t['layer_id']}", f"{'name':<9}{t['layer_name']}",
                          f"{'preset':<9}{t['preset'] if isinstance(t['preset'], str) else '-'}"]
            text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Generation (editor)
    # ------------------------------------------------------------------

    def read_seed(self):
        """The seed in the entry, or None (with a dialog) if it is not a whole number."""
        text = self.seed_var.get().strip()
        try:
            seed = int(text)
            if seed < 0:
                raise ValueError
            return seed
        except ValueError:
            self.error("Seed", f"The seed must be a whole number 0 or larger, not {text!r}.")
            return None

    def new_seed(self):
        seed = int(np.random.default_rng().integers(0, 1_000_000))
        self.seed_var.set(str(seed))
        self.regenerate()

    def regenerate(self):
        """Generate the current recipe and show it."""
        seed = self.read_seed()
        if seed is None:
            return
        try:
            forest = generate(self.recipe, seed=seed)
        except RecipeError as exc:
            self.error("The recipe has problems", "\n\n".join(exc.errors))
            return
        was_viewer = self.mode == "viewer"
        self._set_mode("editor")
        self.forest = forest
        self.status_extra = f" · generated in {forest.elapsed:.2f} s"
        if forest.warnings:
            self.status_extra += " · " + "; ".join(forest.warnings)
        self.show(forest.trees, forest.window, forest.truth,
                  note=window_description(forest.recipe["window"]), source=self.recipe["name"],
                  fit=was_viewer)
        if was_viewer:
            self.hide_var.set(False)
            self._on_hide()

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def open_tree_file(self, path=None):
        """Open a CSV or RDS tree file in the viewer."""
        if path is None:
            path = filedialog.askopenfilename(
                parent=self.root, title="Open a tree file",
                filetypes=[("Tree tables", "*.csv *.rds *.CSV *.RDS"), ("All files", "*.*")])
            if not path:
                return
        self.set_status(f"Reading {os.path.basename(path)} …")
        self.root.update_idletasks()
        try:
            trees = read_trees(path)
            window, note, _ = window_for_trees(path, trees)
        except TreeFileError as exc:
            self.error("Cannot open the file", str(exc))
            self.update_status()
            return
        truth_path = sidecar_paths(path)["truth"]
        truth = read_truth(truth_path, len(trees)) if os.path.exists(truth_path) else None
        self.status_extra = ""
        self.hide_var.set(True)                     # external data: coordinates hidden
        self.map.set_hide_coordinates(True)
        self.shown = None
        self._set_mode("viewer")
        self.show(trees, window, truth, note=note, source=path, fit=True)
        self.update_title()

    def new_recipe(self):
        self.recipe = new_recipe()
        self.recipe_path = None
        self.seed_var.set(str(self.recipe["seed"]))
        self.regenerate()

    def open_recipe(self, path=None):
        from engine.recipe import load_recipe
        if path is None:
            path = filedialog.askopenfilename(parent=self.root, title="Open a recipe",
                                              filetypes=[("Recipes", "*.json"), ("All files", "*.*")])
            if not path:
                return
        try:
            recipe = load_recipe(path)
        except RecipeError as exc:
            self.error("Cannot open the recipe", "\n\n".join(exc.errors))
            return
        self.recipe, self.recipe_path = recipe, path
        self.seed_var.set(str(recipe["seed"]))
        self.regenerate()
        self.map.fit()
        self.update_title()

    def save_recipe(self, path=None):
        from engine.recipe import save_recipe
        if path is None:
            path = filedialog.asksaveasfilename(
                parent=self.root, title="Save recipe", defaultextension=".json",
                initialfile=f"{self.recipe['name']}.json", filetypes=[("Recipes", "*.json")])
            if not path:
                return False
        seed = self.read_seed()
        if seed is None:
            return False
        self.recipe["seed"] = seed
        try:
            save_recipe(self.recipe, path)
        except (RecipeError, OSError) as exc:
            self.error("Cannot save the recipe", str(exc))
            return False
        self.recipe_path = path
        self.update_title()
        self.set_status(f"Saved {path}")
        return True

    def export_dialog(self):
        """Save the map as an image (tables are exported from the editor)."""
        if self.shown is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save the map as an image", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")])
        if not path:
            return
        try:
            self.map.save_image(path)
        except (OSError, ValueError) as exc:
            self.error("Cannot save the image", str(exc))
            return
        self.set_status(f"Saved {path}")

    def quit(self):
        self.root.destroy()


def main(view_file=None, recipe_file=None):
    """Create the window and run the event loop."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:        # older Windows: keep default scaling
            pass
    root = tk.Tk()
    App(root, view_file=view_file, recipe_file=recipe_file)
    root.mainloop()
    return 0
