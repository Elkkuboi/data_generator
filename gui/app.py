"""Main window: viewer and painting editor (tkinter).

Layout: tools and view controls on the left, the map in the centre, layers
and statistics on the right, main buttons and a status bar at the bottom.
The window has two modes:

* editor - paint layers; the forest is generated from the recipe;
* viewer - shows a CSV or RDS file (synthetic or real); coordinates are
  hidden by default because real inventory coordinates are confidential.

Safety: every edit is validated and recorded for undo/redo; the recipe is
autosaved to a recovery file after every change and offered back after a
crash.  Errors never show a traceback: they become dialogs, are written to
a log file, and the program keeps running.
"""

import copy
import datetime
import json
import os
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace

import numpy as np
import shapely
from shapely.geometry import box

from engine.generate import generate
from engine.generators import apply_generator
from engine.geometry import shape_geometry, translate_shape, window_geometry
from engine.io import (
    TreeFileError,
    recipe_for_file,
    read_trees,
    read_truth,
    sidecar_paths,
    window_for_trees,
    write_forest,
)
from engine.presets import PRESETS
from engine.recipe import (
    ADDING_TYPES,
    RecipeError,
    atomic_write_text,
    load_recipe,
    new_recipe,
    is_file_background,
    next_layer_id,
    recipe_to_json,
    save_recipe,
    validate_recipe,
)
from engine.report import forest_report, format_report, window_description

from . import figures, fonts
from .dialogs import SettingsDialog, ask_export, ask_form
from .layers_panel import BACKGROUND, LayersPanel
from .mapview import MapView
from .stats_panel import StatsPanel
from .tools import BrushTool, CircleTool, CorridorTool, PolygonTool, RectangleTool, SelectTool

APP_NAME = "synthforest"
FILTER_DELAY_MS = 250
VIEW_STATS_DELAY_MS = 500
POLL_MS = 40
DRAFT_SCALE = 0.1
MAX_UNDO = 200
TOOLS = (("select", "Select / move"), ("brush", "Brush"), ("circle", "Circle"),
         ("rectangle", "Rectangle"), ("polygon", "Polygon"), ("corridor", "Corridor"))
ACTIONS = (("trees", "Trees (new layer)"), ("clear", "Eraser (clear)"), ("thin", "Thin"))
ACTION_COLOURS = {"trees": "#2ca02c", "clear": "#d62728", "thin": "#9467bd"}


def config_dir():
    """Folder for the recovery file and the error log."""
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, APP_NAME)
    return os.path.join(os.path.expanduser("~"), "." + APP_NAME)


def recovery_path():
    return os.path.join(config_dir(), "recovery.json")


def log_error(text):
    """Append a traceback to the error log; never raises."""
    try:
        os.makedirs(config_dir(), exist_ok=True)
        with open(os.path.join(config_dir(), "errors.log"), "a", encoding="utf-8") as fh:
            fh.write(f"--- {datetime.datetime.now().isoformat(timespec='seconds')}\n{text}\n")
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
        self.bind("<Enter>", lambda e: self._bind_wheel(True))
        self.bind("<Leave>", lambda e: self._bind_wheel(False))

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

    def __init__(self, root, view_file=None, recipe_file=None, edit_file=None, check_recovery=True):
        self.root = root
        self.mode = "editor"
        self.recipe = new_recipe()
        self.recipe_path = None
        self.saved_json = recipe_to_json(self.recipe)
        self.history = [self.saved_json]
        self.history_pos = 0
        self.selected_id = None
        self.shown = None                 # SimpleNamespace(trees, truth, window, note, source)
        self.forest = None                # last generated Forest (editor)
        self.selected_tree = None
        self.status_extra = ""
        self._filter_job = None
        self._view_job = None
        self._gen_thread = None
        self._gen_request = None
        self._gen_result = None
        self._fit_after_generation = False

        root.title(APP_NAME)
        root.geometry("1450x920")
        root.minsize(1000, 650)
        root.report_callback_exception = self._on_tk_error
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self._build()
        self._bind_keys()
        self._make_tools()
        self.layers.refresh()
        self._set_mode("editor")

        if view_file:
            root.after(50, lambda: self.open_tree_file(view_file))
        elif edit_file:
            root.after(50, lambda: self.edit_tree_file(edit_file))
        elif recipe_file:
            root.after(50, lambda: self.open_recipe(recipe_file))
        else:
            root.after(50, lambda: self._startup(check_recovery))

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self):
        style = ttk.Style(self.root)
        if not sys.platform.startswith("win") and "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Status.TLabel", padding=(6, 2))
        style.configure("Head.TLabel", font=fonts.get("head"))

        bottom = ttk.Frame(self.root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(bottom, textvariable=self.status, style="Status.TLabel", anchor="w",
                  relief="sunken").pack(side=tk.BOTTOM, fill=tk.X)
        bar = ttk.Frame(bottom, padding=(4, 4))
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._build_buttons(bar)

        panes = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)
        self.left = ScrollableFrame(panes, width=255)
        self.map = MapView(panes, on_tree_selected=self._on_tree_selected,
                           on_view_changed=self._on_view_changed,
                           on_error=lambda exc: self._on_tk_error(type(exc), exc, exc.__traceback__))
        self.right = ttk.Notebook(panes, width=440)
        panes.add(self.left, weight=0)
        panes.add(self.map.frame, weight=1)
        panes.add(self.right, weight=0)

        self.editor_left = ttk.Frame(self.left.inner)
        self.editor_left.pack(fill=tk.X)
        self._build_tools(self.editor_left)
        self.viewer_left = ttk.Frame(self.left.inner)
        self.viewer_left.pack(fill=tk.X)
        self._build_view_controls(self.viewer_left)

        self.layers_scroll = ScrollableFrame(self.right, width=430)
        self.layers = LayersPanel(self, self.layers_scroll.inner)
        self.layers.frame.pack(fill=tk.BOTH, expand=True)
        self.right.add(self.layers_scroll, text="Layers")
        self.stats = StatsPanel(self.right)

    def _build_buttons(self, bar):
        self.main_buttons = {}

        def button(key, text, command):
            b = ttk.Button(bar, text=text, command=command)
            b.pack(side=tk.LEFT, padx=2)
            self.main_buttons[key] = b
            return b

        def separator():
            ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=6)

        button("new", "New", self.new_recipe)
        button("open_recipe", "Open recipe…", self.open_recipe)
        button("save_recipe", "Save recipe…", self.save_recipe)
        separator()
        button("undo", "Undo", self.undo)
        button("redo", "Redo", self.redo)
        separator()
        ttk.Label(bar, text="Seed").pack(side=tk.LEFT, padx=(2, 2))
        self.seed_var = tk.StringVar(value=str(self.recipe["seed"]))
        self.seed_entry = ttk.Entry(bar, textvariable=self.seed_var, width=9)
        self.seed_entry.pack(side=tk.LEFT)
        self.seed_entry.bind("<Return>", lambda e: self._on_seed_entered())
        button("new_seed", "New seed", self.new_seed)
        button("generate", "Generate", self.generate_full)
        self.draft_var = tk.BooleanVar(value=True)
        self.draft_check = ttk.Checkbutton(bar, text="Draft (10 %, auto)", variable=self.draft_var,
                                           command=self._on_draft)
        self.draft_check.pack(side=tk.LEFT, padx=4)
        separator()
        button("export", "Export…", self.export)
        button("open_file", "Open file…", self.open_tree_file)
        button("edit_file", "Edit this file", self.edit_tree_file)
        button("settings", "Settings…", self.settings)

    def _section(self, parent, title):
        ttk.Label(parent, text=title, style="Head.TLabel").pack(anchor="w", pady=(10, 2), padx=4)
        frame = ttk.Frame(parent, padding=(6, 0))
        frame.pack(fill=tk.X)
        return frame

    def _build_tools(self, parent):
        tools = self._section(parent, "Tools")
        self.tool_var = tk.StringVar(value="select")
        for i, (key, label) in enumerate(TOOLS):
            ttk.Radiobutton(tools, text=label, value=key, variable=self.tool_var,
                            command=self._on_tool).grid(row=i // 2, column=i % 2, sticky="w")
        sizes = ttk.Frame(tools)
        sizes.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(sizes, text="Brush Ø (m)").grid(row=0, column=0, sticky="w")
        self.brush_var = tk.StringVar(value="8")
        ttk.Spinbox(sizes, from_=0.5, to=200, increment=1, textvariable=self.brush_var,
                    width=6).grid(row=0, column=1, padx=4)
        ttk.Label(sizes, text="Corridor width (m)").grid(row=1, column=0, sticky="w")
        self.corridor_var = tk.StringVar(value="4")
        ttk.Spinbox(sizes, from_=0.5, to=200, increment=0.5, textvariable=self.corridor_var,
                    width=6).grid(row=1, column=1, padx=4)

        paint = self._section(parent, "Each new shape becomes")
        self.action_var = tk.StringVar(value="trees")
        for i, (key, label) in enumerate(ACTIONS):
            ttk.Radiobutton(paint, text=label, value=key, variable=self.action_var,
                            command=self._on_action).grid(row=i, column=0, columnspan=2, sticky="w")
        ttk.Label(paint, text="Type").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.new_type_var = tk.StringVar(value="random")
        ttk.Combobox(paint, textvariable=self.new_type_var, values=ADDING_TYPES, state="readonly",
                     width=14).grid(row=3, column=1, sticky="ew", pady=(4, 0))
        ttk.Label(paint, text="Preset").grid(row=4, column=0, sticky="w")
        self.new_preset_var = tk.StringVar(value="(none)")
        preset = ttk.Combobox(paint, textvariable=self.new_preset_var,
                              values=("(none)",) + tuple(PRESETS), state="readonly", width=14)
        preset.grid(row=4, column=1, sticky="ew")
        preset.bind("<<ComboboxSelected>>", lambda e: self._on_new_preset())
        ttk.Label(paint, text="Mode").grid(row=5, column=0, sticky="w")
        self.new_mode_var = tk.StringVar(value="replace")
        ttk.Combobox(paint, textvariable=self.new_mode_var, values=("replace", "add"),
                     state="readonly", width=14).grid(row=5, column=1, sticky="ew")
        self.copy_selected_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(paint, text="Copy settings of the selected layer",
                        variable=self.copy_selected_var).grid(row=6, column=0, columnspan=2, sticky="w")
        ttk.Label(paint, text="Thin: fraction").grid(row=7, column=0, sticky="w", pady=(4, 0))
        self.thin_var = tk.StringVar(value="0.3")
        ttk.Entry(paint, textvariable=self.thin_var, width=8).grid(row=7, column=1, sticky="w", pady=(4, 0))
        paint.columnconfigure(1, weight=1)

        gens = self._section(parent, "Generators (add layers)")
        for i, (key, label) in enumerate((("mosaic", "Mosaic…"), ("gaps", "Gaps…"),
                                          ("trails", "Trails…"), ("understorey", "Understorey…"),
                                          ("beetle_patches", "Beetle patches…"))):
            ttk.Button(gens, text=label, command=lambda k=key: self.run_generator(k)).grid(
                row=i // 2, column=i % 2, sticky="ew", padx=1, pady=1)
        gens.columnconfigure(0, weight=1)
        gens.columnconfigure(1, weight=1)

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
        self.overlay_var = tk.BooleanVar(value=True)
        self.overlay_check = ttk.Checkbutton(view, text="Show layer shapes", variable=self.overlay_var,
                                             command=self.update_overlays)
        self.overlay_check.grid(row=3, column=0, columnspan=2, sticky="w")
        self.view_stats_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(view, text="Statistics for current view only", variable=self.view_stats_var,
                        command=self.update_stats).grid(row=4, column=0, columnspan=2, sticky="w")
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
        self.tree_text = tk.Text(tree, height=14, width=30, font=fonts.get("fixed"),
                                 borderwidth=1, relief="solid")
        self.tree_text.pack(fill=tk.X)
        self._show_tree_info(None)

    def _bind_keys(self):
        bindings = {"<Control-o>": lambda e: self.open_tree_file(),
                    "<Control-s>": lambda e: self.save_recipe(self.recipe_path),
                    "<Control-n>": lambda e: self.new_recipe(),
                    "<Control-z>": lambda e: self.undo(),
                    "<Control-y>": lambda e: self.redo(),
                    "<Control-Z>": lambda e: self.redo(),
                    "<F5>": lambda e: self.generate_full()}
        for sequence, handler in bindings.items():
            self.root.bind_all(sequence, lambda e, h=handler: self._key_command(e, h))

    def _key_command(self, event, handler):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text, ttk.Spinbox)) and \
                event.keysym.lower() in ("z", "y"):
            return None          # let text fields undo their own typing
        handler(event)
        return "break"

    def _make_tools(self):
        on_shape = self.add_shape_layer
        self.tools = {
            "select": SelectTool(self.map, self.pick_layer_at, self.selected_geometry, self.move_selected),
            "brush": BrushTool(self.map, on_shape, lambda: self._size(self.brush_var, 8.0)),
            "circle": CircleTool(self.map, on_shape),
            "rectangle": RectangleTool(self.map, on_shape),
            "polygon": PolygonTool(self.map, on_shape),
            "corridor": CorridorTool(self.map, on_shape, lambda: self._size(self.corridor_var, 4.0)),
        }
        self.map.tool = self.tools["select"]

    # ------------------------------------------------------------------
    # Errors and status
    # ------------------------------------------------------------------

    def _on_tk_error(self, exc_type, exc, tb):
        log_error("".join(traceback.format_exception(exc_type, exc, tb)))
        self.autosave()
        self.error("Something went wrong", f"{exc_type.__name__}: {exc}\n\nYour work is kept; "
                   f"details were written to {os.path.join(config_dir(), 'errors.log')}.")

    def error(self, title, message):
        messagebox.showerror(title, message, parent=self.root)

    def set_status(self, text):
        self.status.set(text)

    def coordinates_hidden(self):
        return self.hide_var.get()

    # ------------------------------------------------------------------
    # Startup, recovery, quitting
    # ------------------------------------------------------------------

    def _startup(self, check_recovery):
        if check_recovery and self._offer_recovery():
            return
        self.request_generation(fit=True)

    def _offer_recovery(self):
        path = recovery_path()
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            recipe = validate_recipe(data["recipe"])
        except (OSError, ValueError, KeyError, TypeError):
            return False
        when = data.get("saved_at", "an earlier session")
        if not messagebox.askyesno(
                "Recover unsaved work",
                f"synthforest did not close normally. Restore the recipe \"{recipe['name']}\" "
                f"autosaved at {when}?", parent=self.root):
            self._remove_recovery()
            return False
        self.recipe_path = data.get("recipe_path")
        self._load_recipe_state(recipe, saved=False)
        self.request_generation(fit=True)
        self.set_status("Recovered the autosaved recipe.")
        return True

    def autosave(self):
        """Write the current recipe to the recovery file (never raises)."""
        try:
            os.makedirs(config_dir(), exist_ok=True)
            data = {"saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "recipe_path": self.recipe_path, "recipe": self.recipe}
            atomic_write_text(recovery_path(), json.dumps(data, indent=1))
        except (OSError, TypeError, ValueError) as exc:
            log_error(f"autosave failed: {exc}")

    def _remove_recovery(self):
        try:
            os.remove(recovery_path())
        except OSError:
            pass

    def unsaved(self):
        return recipe_to_json(self.recipe) != self.saved_json

    def _confirm_discard(self, action):
        """Ask to save unsaved changes; returns False if the user cancels."""
        if not self.unsaved():
            return True
        answer = messagebox.askyesnocancel(
            action, "The recipe has unsaved changes. Save them first?", parent=self.root)
        if answer is None:
            return False
        if answer:
            return self.save_recipe(self.recipe_path)
        return True

    def quit(self):
        if not self._confirm_discard("Quit"):
            return
        self._remove_recovery()
        self.root.destroy()

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------

    def _set_mode(self, mode):
        self.mode = mode
        editing = mode == "editor"
        for key in ("new_seed", "generate", "undo", "redo", "settings", "save_recipe"):
            self.main_buttons[key].configure(state="normal" if editing else "disabled")
        self.seed_entry.configure(state="normal" if editing else "disabled")
        self.draft_check.configure(state="normal" if editing else "disabled")
        self.main_buttons["edit_file"].configure(state="disabled" if editing else "normal")
        if editing:
            if not self.editor_left.winfo_ismapped():
                self.editor_left.pack(fill=tk.X, before=self.viewer_left)
            if str(self.layers_scroll) not in self.right.tabs():
                self.right.insert(0, self.layers_scroll, text="Layers")
        else:
            self.editor_left.pack_forget()
            if str(self.layers_scroll) in self.right.tabs():
                self.right.forget(self.layers_scroll)
            self.tool_var.set("select")
            self._on_tool()
        self.update_overlays()
        self.update_title()

    def update_title(self):
        if self.mode == "viewer" and self.shown is not None:
            self.root.title(f"{APP_NAME} – viewer – {os.path.basename(self.shown.source)}")
        else:
            name = self.recipe.get("name", "forest")
            where = f" ({os.path.basename(self.recipe_path)})" if self.recipe_path else ""
            star = " *" if self.unsaved() else ""
            self.root.title(f"{APP_NAME} – {name}{where}{star}")

    def _ensure_editor(self):
        if self.mode != "editor":
            self._set_mode("editor")
            self._default_hide()

    def _default_hide(self):
        """Hide coordinates by default when the recipe paints on a tree file (maybe real data)."""
        self.hide_var.set(is_file_background(self.recipe["background"]))
        self._on_hide()

    def recipe_dir(self):
        """Folder that relative background file paths are taken from."""
        return os.path.dirname(os.path.abspath(self.recipe_path)) if self.recipe_path else None

    # ------------------------------------------------------------------
    # Recipe editing, undo and redo
    # ------------------------------------------------------------------

    def edit(self, mutate, label="Edit", regenerate=True):
        """Apply ``mutate(recipe_copy)``; validate, record undo, autosave, regenerate.

        Returns True if the edit was applied.  An invalid result shows the
        validation messages and leaves the recipe unchanged.
        """
        candidate = copy.deepcopy(self.recipe)
        try:
            mutate(candidate)
            candidate = validate_recipe(candidate)
        except RecipeError as exc:
            self.error(f"{label}: not applied", "\n\n".join(exc.errors))
            self.layers.refresh()
            return False
        text = recipe_to_json(candidate)
        if text == self.history[self.history_pos]:
            self.layers.refresh()
            return False
        self.recipe = candidate
        del self.history[self.history_pos + 1:]
        self.history.append(text)
        if len(self.history) > MAX_UNDO:
            del self.history[0]
        self.history_pos = len(self.history) - 1
        self._after_change(label, regenerate)
        return True

    def _after_change(self, label, regenerate=True):
        self._ensure_editor()
        if self.selected_id not in (None, BACKGROUND) and not any(
                layer["id"] == self.selected_id for layer in self.recipe["layers"]):
            self.selected_id = None
        self.autosave()
        self.layers.refresh()
        self.update_overlays()
        self.update_title()
        self._update_undo_buttons()
        if not regenerate:
            self.set_status(f"{label}.")
        elif self.draft_var.get():
            self.request_generation()
        else:
            self.status_extra = " · recipe changed: press Generate"
            self.update_status()

    def undo(self):
        if self.mode != "editor" or self.history_pos == 0:
            return
        self.history_pos -= 1
        self.recipe = validate_recipe(json.loads(self.history[self.history_pos]))
        self._after_change("Undo")

    def redo(self):
        if self.mode != "editor" or self.history_pos >= len(self.history) - 1:
            return
        self.history_pos += 1
        self.recipe = validate_recipe(json.loads(self.history[self.history_pos]))
        self._after_change("Redo")

    def _update_undo_buttons(self):
        editing = self.mode == "editor"
        self.main_buttons["undo"].configure(
            state="normal" if editing and self.history_pos > 0 else "disabled")
        self.main_buttons["redo"].configure(
            state="normal" if editing and self.history_pos < len(self.history) - 1 else "disabled")

    def _load_recipe_state(self, recipe, saved=True):
        """Replace the recipe (new, opened or recovered) and reset the history."""
        self.recipe = recipe
        text = recipe_to_json(recipe)
        self.history = [text]
        self.history_pos = 0
        self.saved_json = text if saved else ""
        self.selected_id = None
        self.seed_var.set(str(recipe["seed"]))
        self._ensure_editor()
        self._default_hide()
        self.layers.refresh()
        self.update_overlays()
        self._update_undo_buttons()
        self.update_title()
        self.autosave()

    # ------------------------------------------------------------------
    # Layers: selection, painting, moving
    # ------------------------------------------------------------------

    def select_layer(self, iid):
        self.selected_id = iid
        self.layers.refresh()
        self.update_overlays()

    def set_layer_visible(self, iid, visible):
        def mutate(recipe):
            next(layer for layer in recipe["layers"] if layer["id"] == iid)["visible"] = visible
        self.edit(mutate, "Show layer" if visible else "Hide layer", regenerate=False)

    def _layer_geometries(self):
        window = window_geometry(self.recipe["window"])
        out = []
        for i, layer in enumerate(self.recipe["layers"]):
            try:
                out.append((layer, shape_geometry(layer["shape"], window), i))
            except Exception as exc:      # a broken shape must not break drawing
                log_error(f"shape of layer {layer['id']}: {exc}")
        return out

    def update_overlays(self):
        if self.mode != "editor" or not self.overlay_var.get():
            self.map.set_overlays([])
            return
        items = []
        for layer, geom, i in self._layer_geometries():
            if not layer["visible"] or geom.is_empty or layer["shape"]["kind"] == "window" \
                    and layer["id"] != self.selected_id:
                continue
            if layer["type"] == "clear":
                colour = "#444444"
            elif layer["type"] == "thin":
                colour = "#8e44ad"
            else:
                colour = figures.layer_colour(i + 1)
            items.append((geom, colour, layer["id"] == self.selected_id))
        self.map.set_overlays(items)

    def selected_geometry(self):
        if self.mode != "editor" or self.selected_id in (None, BACKGROUND):
            return None
        for layer, geom, _ in self._layer_geometries():
            if layer["id"] == self.selected_id and layer["shape"]["kind"] != "window":
                return geom
        return None

    def pick_layer_at(self, x, y):
        """Select the topmost visible layer whose shape contains (x, y)."""
        if self.mode != "editor":
            return
        for layer, geom, _ in reversed(self._layer_geometries()):
            if layer["visible"] and layer["shape"]["kind"] != "window" and not geom.is_empty \
                    and shapely.contains_xy(geom, x, y):
                if layer["id"] != self.selected_id:
                    self.select_layer(layer["id"])
                return
        if self.selected_id is not None:
            self.select_layer(None)

    def move_selected(self, dx, dy):
        iid = self.selected_id

        def mutate(recipe):
            layer = next(item for item in recipe["layers"] if item["id"] == iid)
            moved = translate_shape(layer["shape"], dx, dy)
            for key in ("center",):
                if key in moved:
                    moved[key] = [round(v, 2) for v in moved[key]]
            for key in ("vertices", "points"):
                if key in moved:
                    moved[key] = [[round(a, 2), round(b, 2)] for a, b in moved[key]]
            layer["shape"] = moved
        if not self.edit(mutate, "Move shape"):
            self.update_overlays()

    def _size(self, var, default):
        try:
            value = float(var.get().replace(",", "."))
            return value if value > 0 else default
        except ValueError:
            return default

    def add_shape_layer(self, shape):
        """A painting tool finished a shape: turn it into a new layer."""
        action = self.action_var.get()
        new_id = next_layer_id(self.recipe)
        if action == "clear":
            layer = {"name": f"Eraser {new_id}", "type": "clear"}
        elif action == "thin":
            try:
                fraction = float(self.thin_var.get().replace(",", "."))
            except ValueError:
                self.error("Thin", f"The thinning fraction {self.thin_var.get()!r} is not a number.")
                return
            layer = {"name": f"Thin {new_id}", "type": "thin", "params": {"fraction": fraction}}
        else:
            layer = self._new_tree_layer(new_id)
        layer.update(id=new_id, shape=shape)
        if self.edit(lambda r: r["layers"].append(layer), f"Add layer {layer['name']}"):
            self.select_layer(new_id)

    def _new_tree_layer(self, new_id):
        source = None
        if self.copy_selected_var.get() and self.selected_id not in (None,):
            try:
                source = self.layers._get(self.selected_id)
            except StopIteration:
                source = None
        if source is not None and source["type"] in ADDING_TYPES:
            layer = {k: copy.deepcopy(v) for k, v in source.items()
                     if k in ("type", "preset", "params", "composition")}
            layer["mode"] = source.get("mode", self.new_mode_var.get())
            layer["name"] = f"{source.get('name', 'Background')} {new_id}"
            return layer
        preset = self.new_preset_var.get()
        layer = {"type": self.new_type_var.get(), "mode": self.new_mode_var.get()}
        if preset != "(none)":
            layer.update(preset=preset, type=PRESETS[preset]["type"])
        layer["name"] = f"{preset if preset != '(none)' else layer['type']} {new_id}"
        return layer

    def _on_new_preset(self):
        preset = self.new_preset_var.get()
        if preset in PRESETS:
            self.new_type_var.set(PRESETS[preset]["type"])

    def _on_tool(self):
        key = self.tool_var.get()
        if self.map.tool is not None:
            self.map.tool.deactivate()
        if key != "select" and self.map.navigating():
            mode = str(self.map.toolbar.mode)
            if "pan" in mode.lower():
                self.map.toolbar.pan()
            elif "zoom" in mode.lower():
                self.map.toolbar.zoom()
        tool = self.tools[key]
        tool.colour = ACTION_COLOURS[self.action_var.get()]
        self.map.tool = tool
        tool.activate()
        hints = {"select": "Click a tree or a layer; drag the selected layer to move it.",
                 "brush": "Drag to paint a stroke.",
                 "circle": "Press at the centre and drag out the radius.",
                 "rectangle": "Drag from one corner to the opposite corner.",
                 "polygon": "Click the vertices; double-click or Enter to close; Esc cancels.",
                 "corridor": "Click points along the line; double-click or Enter to finish."}
        self.set_status(hints[key])

    def _on_action(self):
        self.map.tool.colour = ACTION_COLOURS[self.action_var.get()]

    # ------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------

    def run_generator(self, name):
        forms = {
            "mosaic": ([("n_patches", "Number of patches", 8, "int")],
                       "Voronoi patches covering the window, each with a random preset "
                       "(replace mode)."),
            "gaps": ([("n_gaps", "Number of gaps", 5, "int"),
                      ("radius_min", "Smallest radius (m)", 10.0, "float"),
                      ("radius_max", "Largest radius (m)", 25.0, "float")],
                     "Circular clearings with small regenerating trees around their edges."),
            "trails": ([("n_trails", "Number of trails", 1, "int"),
                        ("width", "Width (m)", 4.0, "float")],
                       "Corridors crossing the window from edge to edge."),
            "understorey": ([("density", "Density (trees/ha)", 250.0, "float"),
                             ("in_selected", "Only inside the selected layer's shape", False, "bool")],
                            "Small, mostly spruce trees added beneath the canopy (add mode)."),
            "beetle_patches": ([("n_foci", "Number of foci", 3, "int"),
                                ("patches_per_focus", "Patches per focus", 4, "int"),
                                ("dead_share", "Dead share", 0.7, "float")],
                               "Groups of small patches of mature, mostly dead spruce (replace mode)."),
        }
        fields, note = forms[name]
        values = ask_form(self.root, name.replace("_", " ").capitalize(), fields, note)
        if values is None:
            return
        if name == "understorey":
            in_selected = values.pop("in_selected")
            if in_selected:
                geom_layer = next((layer for layer in self.recipe["layers"]
                                   if layer["id"] == self.selected_id), None)
                if geom_layer is None:
                    self.error("Understorey", "Select a layer first, or untick "
                               "\"Only inside the selected layer's shape\".")
                    return
                values["shape"] = geom_layer["shape"]
        rng = np.random.default_rng()
        before = {layer["id"] for layer in self.recipe["layers"]}
        try:
            updated = apply_generator(self.recipe, name, rng, **values)
        except (RecipeError, ValueError) as exc:
            self.error("Generator", str(exc))
            return
        added = [layer for layer in updated["layers"] if layer["id"] not in before]
        if not added:
            self.error("Generator", "Nothing was added (the window may be too small for these settings).")
            return
        if self.edit(lambda r: r["layers"].extend(added), f"Add {len(added)} {name} layers"):
            self.select_layer(added[-1]["id"])

    # ------------------------------------------------------------------
    # Generation (background thread)
    # ------------------------------------------------------------------

    def read_seed(self, quiet=False):
        """The seed in the entry, or None (with a dialog) if it is not a whole number."""
        text = self.seed_var.get().strip()
        try:
            seed = int(text)
            if not 0 <= seed < 2 ** 63:
                raise ValueError
            return seed
        except ValueError:
            if not quiet:
                self.error("Seed", f"The seed must be a whole number 0 or larger, not {text!r}.")
            return None

    def _on_seed_entered(self):
        seed = self.read_seed()
        if seed is not None:
            self.edit(lambda r: r.__setitem__("seed", seed), "Change seed")
            if not self.draft_var.get():
                self.generate_full()

    def new_seed(self):
        seed = int(np.random.default_rng().integers(0, 1_000_000))
        self.seed_var.set(str(seed))
        self.edit(lambda r: r.__setitem__("seed", seed), "New seed")
        if not self.draft_var.get():
            self.generate_full()

    def _on_draft(self):
        self.request_generation()

    def generate_full(self):
        """Generate at full density (the Generate button)."""
        if self.read_seed() is None:
            return
        self._ensure_editor()
        self.request_generation(scale=1.0)

    def request_generation(self, scale=None, fit=False):
        """Queue a generation of the current recipe; the newest request wins."""
        seed = self.read_seed(quiet=True)
        if seed is None:
            seed = self.recipe["seed"]
        if scale is None:
            scale = DRAFT_SCALE if self.draft_var.get() else 1.0
        self._gen_request = (copy.deepcopy(self.recipe), seed, scale, self.recipe_dir())
        self._fit_after_generation = self._fit_after_generation or fit
        if self._gen_thread is None:
            self._start_generation()

    def _start_generation(self):
        recipe, seed, scale, base_dir = self._gen_request
        self._gen_request = None
        self.set_status(f"Generating{' draft' if scale < 1 else ''} …")

        def work():
            try:
                self._gen_result = ("ok", generate(recipe, seed=seed, density_scale=scale,
                                                   base_dir=base_dir), scale)
            except RecipeError as exc:
                self._gen_result = ("recipe", exc, scale)
            except Exception:              # reported on the main thread
                self._gen_result = ("error", traceback.format_exc(), scale)

        self._gen_result = None
        self._gen_thread = threading.Thread(target=work, daemon=True)
        self._gen_thread.start()
        self.root.after(POLL_MS, self._poll_generation)

    def _poll_generation(self):
        if self._gen_thread is not None and self._gen_thread.is_alive():
            self.root.after(POLL_MS, self._poll_generation)
            return
        self._gen_thread = None
        result = self._gen_result
        if self._gen_request is not None:          # a newer request is waiting
            self._start_generation()
            return
        kind, value, scale = result
        if kind == "recipe":
            self.error("The recipe has problems", "\n\n".join(value.errors))
        elif kind == "error":
            log_error(value)
            self.error("Generation failed", value.strip().splitlines()[-1] +
                       f"\n\nDetails were written to {os.path.join(config_dir(), 'errors.log')}.")
        else:
            self._show_forest(value, scale)

    def _show_forest(self, forest, scale):
        if self.mode != "editor":
            return
        self.forest = forest
        draft = f"draft {int(scale * 100)} % · " if scale < 1 else ""
        self.status_extra = f" · {draft}generated in {forest.elapsed:.2f} s"
        if forest.warnings:
            self.status_extra += " · " + "; ".join(forest.warnings)
        fit, self._fit_after_generation = self._fit_after_generation, False
        self.show(forest.trees, forest.window, forest.truth,
                  window_spec=forest.recipe["window"], source=self.recipe["name"],
                  fit=fit or self.shown is None)

    # ------------------------------------------------------------------
    # Showing data, filter, statistics
    # ------------------------------------------------------------------

    def show(self, trees, window, truth=None, window_spec=None, origin="", source="", fit=False):
        """Display a tree table (generated or read from a file)."""
        self.shown = SimpleNamespace(trees=trees, truth=truth, window=window,
                                     window_spec=window_spec, origin=origin, source=source)
        self.map.set_forest(trees, window, truth, fit=fit)
        if self.colour_var.get() == "layer" and truth is None:
            self.colour_var.set("species")
        self.map.colour_by = self.colour_var.get()
        self.apply_filter()

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
        self.map.set_visible(np.flatnonzero(self.visible_mask()))
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
        text = f"Scope: {scope}; {len(trees):,} of {len(self.shown.trees):,} trees pass the filter."
        if self.mode == "editor" and self.forest is not None and self.forest.density_scale < 1:
            text += "\nDRAFT at 10 % density: press Generate for the full forest."
        self.stats.update(trees.reset_index(drop=True), region, self.window_note(), text)

    def update_status(self):
        if self.shown is None:
            self.set_status("No trees.")
            return
        n, shown = len(self.shown.trees), len(self.map.visible)
        area = self.shown.window.area / 1e4
        density = n / area if area > 0 else float("nan")
        part = f" ({shown:,} shown)" if shown != n else ""
        prefix = os.path.basename(self.shown.source) if self.mode == "viewer" else self.recipe["name"]
        self.set_status(f"{prefix}: {n:,} trees{part} · {density:.0f} /ha · {self.window_note()}"
                        f"{self.status_extra}")

    def _on_colour(self):
        value = self.colour_var.get()
        if value == "layer" and (self.shown is None or self.shown.truth is None):
            self.error("Colour by layer", "Layer colours need the ground truth of a synthetic "
                       "forest (the <name>-truth.csv file next to the tree file).")
            self.colour_var.set(self.map.colour_by)
            return
        self.map.set_colour_by(value)

    def window_note(self):
        """Window description for the status bar and report; no coordinates when hidden."""
        if self.shown is None or self.shown.window_spec is None:
            return ""
        text = window_description(self.shown.window_spec, hide_centre=self.hide_var.get())
        return f"{text} ({self.shown.origin})" if self.shown.origin else text

    def _on_hide(self):
        self.map.set_hide_coordinates(self.hide_var.get())
        self._show_tree_info(self.selected_tree)
        if self.shown is not None:
            self.update_status()
            self.update_stats()
        if self.mode == "editor":
            self.layers.build_editor()

    def _on_tree_selected(self, index):
        self._show_tree_info(index)

    def _show_tree_info(self, index):
        self.selected_tree = index
        text = self.tree_text
        text.configure(state="normal")
        text.delete("1.0", tk.END)
        if index is None or self.shown is None or index >= len(self.shown.trees):
            text.insert("1.0", "Click a tree on the map\n(Select tool, no pan/zoom).")
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
                preset = t["preset"] if isinstance(t["preset"], str) and t["preset"] != "NA" else "-"
                lines += ["", f"{'layer':<9}{t['layer_id']}", f"{'name':<9}{t['layer_name']}",
                          f"{'preset':<9}{preset}"]
            text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

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
            window, spec, origin, _ = window_for_trees(path, trees)
        except TreeFileError as exc:
            self.error("Cannot open the file", str(exc))
            self.update_status()
            return
        truth_path = sidecar_paths(path)["truth"]
        truth = read_truth(truth_path, len(trees)) if os.path.exists(truth_path) else None
        self.status_extra = ""
        self.hide_var.set(True)                     # external data: coordinates hidden
        self.map.set_hide_coordinates(True)
        self._set_mode("viewer")
        self.show(trees, window, truth, window_spec=spec, origin=origin, source=path, fit=True)
        self.update_title()

    def edit_tree_file(self, path=None):
        """Paint on a tree file: start a new recipe whose background is the file's trees."""
        if path is None and self.mode == "viewer" and self.shown is not None:
            path = self.shown.source
        if path is None:
            path = filedialog.askopenfilename(
                parent=self.root, title="Open a tree file to edit",
                filetypes=[("Tree tables", "*.csv *.rds *.CSV *.RDS"), ("All files", "*.*")])
            if not path:
                return
        if not self._confirm_discard("Edit a tree file"):
            return
        try:
            same = self.shown is not None and self.mode == "viewer" and self.shown.source == path
            trees = self.shown.trees if same else read_trees(path)
            recipe = recipe_for_file(path, trees)
        except (TreeFileError, RecipeError) as exc:
            self.error("Cannot edit the file", str(exc))
            return
        self.recipe_path = None
        self._load_recipe_state(recipe)
        self.request_generation(fit=True)
        self.set_status(f"Editing {os.path.basename(path)}: paint layers on its trees; "
                        "the file itself is never changed.")

    def new_recipe(self):
        if not self._confirm_discard("New recipe"):
            return
        self.recipe_path = None
        self._load_recipe_state(new_recipe())
        self.request_generation(fit=True)

    def open_recipe(self, path=None):
        if not self._confirm_discard("Open recipe"):
            return
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
        self.recipe_path = path
        self._load_recipe_state(recipe)
        self.request_generation(fit=True)

    def save_recipe(self, path=None):
        if self.mode != "editor":
            return False
        if path is None:
            path = filedialog.asksaveasfilename(
                parent=self.root, title="Save recipe", defaultextension=".json",
                initialfile=f"{self.recipe['name']}.json", filetypes=[("Recipes", "*.json")])
            if not path:
                return False
        seed = self.read_seed()
        if seed is None:
            return False
        if seed != self.recipe["seed"]:
            self.edit(lambda r: r.__setitem__("seed", seed), "Change seed")
        try:
            save_recipe(self.recipe, path)
        except (RecipeError, OSError) as exc:
            self.error("Cannot save the recipe", str(exc))
            return False
        self.recipe_path = path
        self.saved_json = recipe_to_json(self.recipe)
        self.update_title()
        self.set_status(f"Saved {path}")
        return True

    def export(self):
        """Export tables, truth, recipe, report and/or the map image."""
        if self.shown is None:
            return
        if self.mode == "viewer":
            self._export_image_only()
            return
        default = os.path.join(os.path.dirname(self.recipe_path) if self.recipe_path else os.getcwd(),
                               self.recipe["name"])
        options = ask_export(self.root, default, self.hide_var.get())
        if options is None:
            return
        seed = self.read_seed()
        if seed is None:
            return
        self.set_status("Generating the full forest for export …")
        self.root.update_idletasks()
        try:
            forest = generate(self.recipe, seed=seed, base_dir=self.recipe_dir())
        except RecipeError as exc:
            self.error("The recipe has problems", "\n\n".join(exc.errors))
            return
        base = options["base"]
        self._show_forest(forest, 1.0)          # the image shows the exported forest
        try:
            tables = [fmt for fmt in ("csv", "rds") if options[fmt]]
            report = None
            if options["report"]:
                report = format_report(
                    forest_report(forest.trees, forest.window,
                                  window_description(forest.recipe["window"],
                                                     hide_centre=self.hide_var.get())),
                    title=f"{os.path.basename(base)} (seed {forest.recipe['seed']})")
            written = write_forest(forest, base, tables, report_text=report,
                                   truth=options["truth"], recipe=options["recipe"])
            if options["image"]:
                path = f"{base}.{options['image_format']}"
                self.map.save_image(path)
                written.append(path)
        except (OSError, ValueError) as exc:
            self.error("Export failed", str(exc))
            return
        messagebox.showinfo("Export", "Wrote:\n" + "\n".join(written), parent=self.root)
        self.set_status(f"Exported {len(written)} files to {os.path.dirname(os.path.abspath(base))}")

    def _export_image_only(self):
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

    def settings(self):
        if self.mode != "editor":
            return

        def apply(changes):
            def mutate(recipe):
                recipe["name"] = changes["name"] or recipe["name"]
                recipe["window"] = changes["window"]
                recipe["laser_artefacts"] = changes["laser_artefacts"]
                recipe["background"]["enabled"] = changes["background"]
            ok = self.edit(mutate, "Change settings")
            if ok:
                self._fit_after_generation = True
                self.request_generation(fit=True)
            return ok
        SettingsDialog(self.root, self.recipe, apply, hide_coordinates=self.hide_var.get())


def main(view_file=None, recipe_file=None, edit_file=None):
    """Create the window and run the event loop."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:        # older Windows: keep default scaling
            pass
    root = tk.Tk()
    App(root, view_file=view_file, recipe_file=recipe_file, edit_file=edit_file)
    root.mainloop()
    return 0
