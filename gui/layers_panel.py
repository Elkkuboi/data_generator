"""Layer list and parameter editor.

The list shows the background and every layer in order (first = applied
first).  Selecting a row highlights its shape on the map and shows its
parameters below.  Every change goes through ``app.edit``, which validates
the recipe, records an undo step and autosaves; an invalid value shows a
dialog and the editor falls back to the last valid recipe.
"""

import copy
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk

from engine.presets import PRESETS, SIZE_CLASSES, SPECIES
from engine.recipe import (
    ADDING_TYPES,
    LAYER_TYPES,
    PARAM_RANGES,
    is_file_background,
    merged_composition,
    merged_params,
    next_layer_id,
    resolve_layer,
)

from . import fonts

BACKGROUND = "background"
NO_PRESET = "(none)"
DEFAULT_SIZE = "(default)"
CUSTOM_SIZE = "custom dbh"
CHECK = {True: "[x]", False: "[ ]"}
EYE = {True: "yes", False: "no"}
SHAPE_FIELDS = {"circle": ("radius",), "rectangle": ("width", "height"), "corridor": ("width",)}
PARAM_LABELS = {
    "density": "Density (trees/ha)", "cluster_density": "Clusters per ha",
    "trees_per_cluster": "Trees per cluster", "cluster_radius": "Cluster radius (sigma, m)",
    "min_distance": "Min. distance (m)", "row_spacing": "Row spacing (m)",
    "tree_spacing": "Tree spacing (m)", "angle": "Row angle (°)", "jitter": "Jitter (m)",
    "density_start": "Density at start (/ha)", "density_end": "Density at end (/ha)",
    "direction": "Direction (°)", "fraction": "Fraction removed", "dbh_below": "Only dbh below (cm)",
    "dbh_above": "Only dbh above (cm)",
}


def _fmt(value):
    if value is None:
        return ""
    return f"{value:g}" if isinstance(value, float) else str(value)


def balance_shares(shares, index, value):
    """Set share ``index`` to ``value`` and rescale the others so all sum to 1."""
    value = min(max(float(value), 0.0), 1.0)
    others = [i for i in range(len(shares)) if i != index]
    rest = sum(shares[i] for i in others)
    out = list(shares)
    out[index] = value
    for i in others:
        out[i] = (shares[i] / rest * (1.0 - value)) if rest > 1e-9 else (1.0 - value) / len(others)
    out = [round(v, 3) for v in out]
    fix = next((i for i in others if out[i] > 0), others[-1])
    out[fix] = round(1.0 - sum(out[i] for i in range(len(out)) if i != fix), 3)
    if out[fix] < 0:
        out[index] += out[fix]
        out[fix] = 0.0
    return out


class LayersPanel:
    """The "Layers" tab: ordered list, buttons and the parameter editor."""

    def __init__(self, app, parent):
        self.app = app
        self.frame = ttk.Frame(parent)
        self._balancing = False

        top = ttk.Frame(self.frame)
        top.pack(fill=tk.X, padx=4, pady=4)
        self.tree = ttk.Treeview(top, columns=("on", "vis", "name", "type"), show="headings",
                                 height=9, selectmode="browse")
        for col, text, width in (("on", "On", 34), ("vis", "Show", 42), ("name", "Name", 190),
                                 ("type", "Type", 120)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, stretch=col == "name",
                             anchor="center" if col in ("on", "vis") else "w")
        bar = ttk.Scrollbar(top, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Button-1>", self._on_click, add=True)
        self.tree.bind("<Double-1>", lambda e: self.rename())
        self.tree.bind("<Delete>", lambda e: self.delete())

        buttons = ttk.Frame(self.frame)
        buttons.pack(fill=tk.X, padx=4)
        for text, command in (("Up", self.move_up), ("Down", self.move_down),
                              ("Duplicate", self.duplicate), ("Delete", self.delete),
                              ("Rename", self.rename)):
            ttk.Button(buttons, text=text, command=command).pack(side=tk.LEFT, padx=1)

        ttk.Separator(self.frame).pack(fill=tk.X, pady=6)
        self.editor = ttk.Frame(self.frame, padding=(6, 0))
        self.editor.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def refresh(self):
        """Rebuild the list and the editor from ``app.recipe``."""
        recipe = self.app.recipe
        self._refreshing = True
        try:
            self._fill_list(recipe)
        finally:
            self._refreshing = False
        self.build_editor()

    def _fill_list(self, recipe):
        self.tree.delete(*self.tree.get_children())
        bg = recipe["background"]
        name = (f"Trees from {os.path.basename(bg['file'])}" if is_file_background(bg)
                else "Background (whole window)")
        self.tree.insert("", "end", iid=BACKGROUND, values=(
            CHECK[bg["enabled"]], "", name, self._type_text(bg)))
        for layer in recipe["layers"]:
            self.tree.insert("", "end", iid=layer["id"], values=(
                CHECK[layer["enabled"]], EYE[layer["visible"]], layer["name"], self._type_text(layer)))
        iid = self.app.selected_id
        if iid is not None and self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.see(iid)

    @staticmethod
    def _type_text(layer):
        if is_file_background(layer):
            return "file (unchanged)"
        text = layer["type"]
        if layer.get("preset"):
            text = layer["preset"]
        if layer["type"] in ADDING_TYPES and layer.get("mode") == "replace":
            text += " (replace)"
        return text

    def _on_select(self, event=None):
        if getattr(self, "_refreshing", False):
            return
        selection = self.tree.selection()
        iid = selection[0] if selection else None
        if iid != self.app.selected_id:
            self.app.select_layer(iid)

    def _on_click(self, event):
        if self.tree.identify_region(event.x, event.y) != "cell":
            return
        column = self.tree.identify_column(event.x)
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        if column == "#1":
            self.app.edit(lambda r: self._toggle(r, iid, "enabled"),
                          f"{'Disable' if self._get(iid)['enabled'] else 'Enable'} layer")
            return "break"
        if column == "#2" and iid != BACKGROUND:
            self.app.set_layer_visible(iid, not self._get(iid)["visible"])
            return "break"

    def _get(self, iid, recipe=None):
        recipe = recipe or self.app.recipe
        if iid == BACKGROUND:
            return recipe["background"]
        return next(layer for layer in recipe["layers"] if layer["id"] == iid)

    def _index(self, iid, recipe=None):
        recipe = recipe or self.app.recipe
        return next(i for i, layer in enumerate(recipe["layers"]) if layer["id"] == iid)

    def _toggle(self, recipe, iid, key):
        target = self._get(iid, recipe)
        target[key] = not target[key]

    def _selected_layer_id(self):
        iid = self.app.selected_id
        return iid if iid not in (None, BACKGROUND) else None

    def move_up(self):
        self._move(-1)

    def move_down(self):
        self._move(1)

    def _move(self, step):
        iid = self._selected_layer_id()
        if iid is None:
            return

        def mutate(recipe):
            layers = recipe["layers"]
            i = self._index(iid, recipe)
            j = min(max(i + step, 0), len(layers) - 1)
            layers.insert(j, layers.pop(i))
        self.app.edit(mutate, "Move layer")

    def duplicate(self):
        iid = self._selected_layer_id()
        if iid is None:
            return
        new_id = next_layer_id(self.app.recipe)

        def mutate(recipe):
            i = self._index(iid, recipe)
            copy_ = copy.deepcopy(recipe["layers"][i])
            copy_["id"] = new_id
            copy_["name"] = copy_["name"] + " (copy)"
            recipe["layers"].insert(i + 1, copy_)
        if self.app.edit(mutate, "Duplicate layer"):
            self.app.select_layer(new_id)

    def delete(self):
        iid = self._selected_layer_id()
        if iid is None:
            return
        index = self._index(iid)

        def mutate(recipe):
            recipe["layers"].pop(self._index(iid, recipe))
        if self.app.edit(mutate, "Delete layer"):
            layers = self.app.recipe["layers"]
            self.app.select_layer(layers[min(index, len(layers) - 1)]["id"] if layers else None)

    def rename(self):
        iid = self._selected_layer_id()
        if iid is None:
            return
        name = simpledialog.askstring("Rename layer", "New name:", initialvalue=self._get(iid)["name"],
                                      parent=self.frame)
        if name and name.strip():
            self.app.edit(lambda r: self._get(iid, r).__setitem__("name", name.strip()), "Rename layer")

    # ------------------------------------------------------------------
    # Parameter editor
    # ------------------------------------------------------------------

    def build_editor(self):
        for child in self.editor.winfo_children():
            child.destroy()
        iid = self.app.selected_id
        if iid is None:
            ttk.Label(self.editor, text="Select a layer to edit it, or paint a new one\n"
                      "with a tool on the left.", foreground="#555555").pack(anchor="w")
            return
        try:
            layer = self._get(iid)
        except StopIteration:
            return
        self._row = 0
        grid = ttk.Frame(self.editor)
        grid.pack(fill=tk.X)
        grid.columnconfigure(1, weight=1)
        self.grid = grid
        is_bg = iid == BACKGROUND
        if is_bg and is_file_background(layer):
            self._file_background_editor(layer)
            return
        if is_bg:
            ttk.Button(grid, text="Use trees from a file instead…",
                       command=self.choose_background_file).grid(
                row=self._next_row(), column=0, columnspan=3, sticky="w", pady=(0, 6))
        types = ADDING_TYPES if is_bg else LAYER_TYPES
        self._combo("Type", layer["type"], types, lambda v: self._set_type(iid, v))
        if layer["type"] in ADDING_TYPES:
            if not is_bg:
                self._combo("Mode", layer.get("mode", "add"), ("add", "replace"),
                            lambda v: self._set_key(iid, "mode", v))
            self._combo("Preset", layer.get("preset") or NO_PRESET, (NO_PRESET,) + tuple(PRESETS),
                        lambda v: self._set_preset(iid, v))
        self._heading("Pattern" if layer["type"] in ADDING_TYPES else "Parameters")
        effective = merged_params(layer)
        for key, (lo, hi, unit) in PARAM_RANGES[layer["type"]].items():
            explicit = key in layer.get("params", {})
            self._entry(PARAM_LABELS.get(key, key), effective.get(key),
                        lambda text, k=key: self._set_param(iid, k, text), explicit)
        if layer["type"] in ADDING_TYPES:
            self._composition_editor(iid, layer)
        if not is_bg:
            self._shape_editor(iid, layer)
        ttk.Label(self.editor, text="Bold labels: set in this layer; others come from the "
                  "preset or defaults.\nPress Enter to apply a typed value.",
                  foreground="#666666", font=fonts.get("small")).pack(anchor="w", pady=(8, 0))

    def _next_row(self):
        self._row += 1
        return self._row

    # --- background from a file ---------------------------------------

    def _file_background_editor(self, background):
        self._heading("Trees from a file")
        path = background["file"]
        count = self.app.forest.layer_counts.get(BACKGROUND) if self.app.forest else None
        lines = [os.path.basename(path), os.path.dirname(path)]
        if count is not None and self.app.forest.density_scale == 1:
            lines.append(f"{count:,} of its trees are in the forest now.")
        ttk.Label(self.grid, text="\n".join(lines), wraplength=380).grid(
            row=self._next_row(), column=0, columnspan=3, sticky="w")
        ttk.Label(self.grid, foreground="#555555", wraplength=380, text=(
            "The file is only read, never changed. Its trees keep all their values. "
            "Layers below can remove them (Eraser, Thin, replace mode) and add new trees "
            "among them.")).grid(row=self._next_row(), column=0, columnspan=3, sticky="w", pady=4)
        buttons = ttk.Frame(self.grid)
        buttons.grid(row=self._next_row(), column=0, columnspan=3, sticky="w", pady=4)
        ttk.Button(buttons, text="Choose another file…",
                   command=self.choose_background_file).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(buttons, text="Use a generated background",
                   command=self.use_generated_background).pack(side=tk.LEFT)

    def choose_background_file(self):
        path = filedialog.askopenfilename(
            parent=self.frame, title="Trees for the background",
            filetypes=[("Tree tables", "*.csv *.rds *.CSV *.RDS"), ("All files", "*.*")])
        if not path:
            return

        def mutate(recipe):
            recipe["background"] = {"enabled": True, "file": os.path.abspath(path)}
        if self.app.edit(mutate, "Use trees from a file"):
            self.app._default_hide()

    def use_generated_background(self):
        def mutate(recipe):
            recipe["background"] = {"enabled": True, "type": "random", "params": {}, "composition": {}}
        self.app.edit(mutate, "Use a generated background")

    def _heading(self, text):
        ttk.Label(self.grid, text=text, style="Head.TLabel").grid(
            row=self._next_row(), column=0, columnspan=3, sticky="w", pady=(8, 2))

    def _combo(self, label, value, values, on_change):
        row = self._next_row()
        ttk.Label(self.grid, text=label).grid(row=row, column=0, sticky="w")
        var = tk.StringVar(value=value)
        combo = ttk.Combobox(self.grid, textvariable=var, values=list(values), state="readonly")
        combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=1)
        combo.bind("<<ComboboxSelected>>", lambda e: on_change(var.get()))
        return combo

    def _entry(self, label, value, on_commit, explicit=False, width=10):
        row = self._next_row()
        font = fonts.get("bold") if explicit else "TkDefaultFont"
        ttk.Label(self.grid, text=label, font=font).grid(row=row, column=0, sticky="w")
        var = tk.StringVar(value=_fmt(value))
        entry = ttk.Entry(self.grid, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="w", pady=1)
        state = {"last": var.get()}

        def commit(event=None):
            text = var.get()
            if text != state["last"]:        # Enter and FocusOut must not apply twice
                state["last"] = text
                on_commit(text)
        entry.bind("<Return>", commit)
        entry.bind("<KP_Enter>", commit)
        entry.bind("<FocusOut>", commit)
        return entry

    def _number(self, text, allow_blank=False):
        text = text.strip().replace(",", ".")
        if not text and allow_blank:
            return None
        try:
            return float(text)
        except ValueError:
            self.app.error("Not a number", f"{text!r} is not a number.")
            self.frame.after_idle(self.build_editor)
            raise

    # --- edits --------------------------------------------------------

    def _set_type(self, iid, new_type):
        def mutate(recipe):
            layer = self._get(iid, recipe)
            if layer["type"] == new_type:
                return
            layer["type"] = new_type
            layer["params"] = {}
            if new_type in ADDING_TYPES:
                layer.setdefault("composition", {})
                if iid != BACKGROUND:
                    layer.setdefault("mode", "replace")
                if layer.get("preset") and PRESETS[layer["preset"]]["type"] != new_type:
                    layer.pop("preset")
            else:
                for key in ("composition", "mode", "preset"):
                    layer.pop(key, None)
        self.app.edit(mutate, "Change layer type")

    def _set_key(self, iid, key, value):
        self.app.edit(lambda r: self._get(iid, r).__setitem__(key, value), f"Change {key}")

    def _set_preset(self, iid, value):
        def mutate(recipe):
            layer = self._get(iid, recipe)
            if value == NO_PRESET:
                layer.pop("preset", None)
            else:
                layer["preset"] = value
                layer["type"] = PRESETS[value]["type"]
                layer["params"] = {}
                layer["composition"] = {}
        self.app.edit(mutate, "Change preset")

    def _set_param(self, iid, key, text):
        try:
            value = self._number(text, allow_blank=True)
        except ValueError:
            return

        def mutate(recipe):
            params = self._get(iid, recipe).setdefault("params", {})
            if value is None:
                params.pop(key, None)
            else:
                params[key] = value
        self.app.edit(mutate, f"Change {key}")

    # --- composition --------------------------------------------------

    def _composition_editor(self, iid, layer):
        self._heading("Composition")
        comp = merged_composition(layer)
        shares = list(resolve_layer(layer).composition.shares)
        explicit = "species" in layer.get("composition", {})
        self.share_vars = []
        self.share_labels = []
        for i, name in enumerate(SPECIES):
            row = self._next_row()
            ttk.Label(self.grid, text=name, font=fonts.get("bold") if explicit
                      else "TkDefaultFont").grid(row=row, column=0, sticky="w")
            var = tk.DoubleVar(value=shares[i])
            scale = ttk.Scale(self.grid, from_=0.0, to=1.0, variable=var,
                              command=lambda v, k=i: self._on_share(k))
            scale.grid(row=row, column=1, sticky="ew")
            label = ttk.Label(self.grid, text=f"{shares[i]:.2f}", width=5)
            label.grid(row=row, column=2, sticky="e")
            scale.bind("<ButtonRelease-1>", lambda e: self._commit_shares(iid))
            scale.bind("<KeyRelease>", lambda e: self._commit_shares(iid))
            self.share_vars.append(var)
            self.share_labels.append(label)
        if explicit:
            row = self._next_row()
            ttk.Button(self.grid, text="Use preset/default shares",
                       command=lambda: self._clear_comp(iid, "species")).grid(
                row=row, column=1, sticky="w", pady=2)

        if "dbh" in comp:
            size = CUSTOM_SIZE
        else:
            size = comp.get("size_class", DEFAULT_SIZE)
        self._combo("Size class", size, (DEFAULT_SIZE,) + tuple(SIZE_CLASSES) + (CUSTOM_SIZE,),
                    lambda v: self._set_size(iid, v))
        if size == CUSTOM_SIZE:
            dbh = comp["dbh"]
            self._entry("Median dbh (cm)", dbh["median"],
                        lambda t: self._set_dbh(iid, "median", t), True)
            self._entry("dbh spread (log sigma)", dbh["sigma"],
                        lambda t: self._set_dbh(iid, "sigma", t), True)
        dead = comp.get("dead_share")
        self._entry("Dead share (blank = species default)", dead,
                    lambda t: self._set_dead(iid, t), "dead_share" in layer.get("composition", {}))

    def _on_share(self, index):
        if self._balancing:
            return
        self._balancing = True
        try:
            shares = [v.get() for v in self.share_vars]
            balanced = balance_shares(shares, index, shares[index])
            for var, label, value in zip(self.share_vars, self.share_labels, balanced):
                var.set(value)
                label.configure(text=f"{value:.2f}")
        finally:
            self._balancing = False

    def _commit_shares(self, iid):
        shares = [round(v.get(), 3) for v in self.share_vars]
        total = sum(shares)
        if total <= 0:
            return
        values = {name: round(s / total, 3) for name, s in zip(SPECIES, shares)}
        values[SPECIES[-1]] = round(1.0 - values[SPECIES[0]] - values[SPECIES[1]], 3)

        def mutate(recipe):
            self._get(iid, recipe).setdefault("composition", {})["species"] = values
        current = self._get(iid).get("composition", {}).get("species")
        if current != values:
            self.app.edit(mutate, "Change species shares")

    def _clear_comp(self, iid, key):
        self.app.edit(lambda r: self._get(iid, r).setdefault("composition", {}).pop(key, None),
                      f"Reset {key}")

    def _set_size(self, iid, value):
        def mutate(recipe):
            comp = self._get(iid, recipe).setdefault("composition", {})
            comp.pop("size_class", None)
            comp.pop("dbh", None)
            if value == CUSTOM_SIZE:
                median, sigma = SIZE_CLASSES["middle"]
                comp["dbh"] = {"median": median, "sigma": sigma}
            elif value != DEFAULT_SIZE:
                comp["size_class"] = value
        self.app.edit(mutate, "Change size class")

    def _set_dbh(self, iid, key, text):
        try:
            value = self._number(text)
        except ValueError:
            return

        def mutate(recipe):
            comp = self._get(iid, recipe).setdefault("composition", {})
            dbh = dict(comp.get("dbh") or merged_composition(self._get(iid, recipe)).get("dbh"))
            dbh[key] = value
            comp.pop("size_class", None)
            comp["dbh"] = dbh
        self.app.edit(mutate, "Change dbh distribution")

    def _set_dead(self, iid, text):
        try:
            value = self._number(text, allow_blank=True)
        except ValueError:
            return

        def mutate(recipe):
            comp = self._get(iid, recipe).setdefault("composition", {})
            if value is None:
                comp.pop("dead_share", None)
            else:
                comp["dead_share"] = value
        self.app.edit(mutate, "Change dead share")

    # --- shape --------------------------------------------------------

    def _shape_editor(self, iid, layer):
        shape = layer["shape"]
        self._heading(f"Shape: {shape['kind']}")
        for key in SHAPE_FIELDS.get(shape["kind"], ()):
            self._entry(f"{key.capitalize()} (m)", shape[key],
                        lambda t, k=key: self._set_shape_value(iid, k, t), True)
        if shape["kind"] in ("circle", "rectangle"):
            cx, cy = shape["center"]
            hidden = self.app.coordinates_hidden()
            self._entry("Centre x (m)", "hidden" if hidden else cx,
                        lambda t: self._set_centre(iid, 0, t), True)
            self._entry("Centre y (m)", "hidden" if hidden else cy,
                        lambda t: self._set_centre(iid, 1, t), True)
        elif shape["kind"] in ("polygon", "corridor"):
            count = len(shape.get("vertices") or shape.get("points"))
            row = self._next_row()
            ttk.Label(self.grid, text=f"{count} points; drag the shape on the map to move it.",
                      foreground="#555555").grid(row=row, column=0, columnspan=3, sticky="w")

    def _set_shape_value(self, iid, key, text):
        try:
            value = self._number(text)
        except ValueError:
            return
        self.app.edit(lambda r: self._get(iid, r)["shape"].__setitem__(key, value), f"Change {key}")

    def _set_centre(self, iid, axis, text):
        try:
            value = self._number(text)
        except ValueError:
            return

        def mutate(recipe):
            centre = list(self._get(iid, recipe)["shape"]["center"])
            centre[axis] = value
            self._get(iid, recipe)["shape"]["center"] = centre
        self.app.edit(mutate, "Move shape")
