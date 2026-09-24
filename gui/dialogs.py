"""Modal dialogs: a small form helper, Export and Settings."""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import shapely.geometry

from . import fonts


class FormDialog(tk.Toplevel):
    """A modal form of labelled fields; ``result`` is a dict or None if cancelled.

    ``fields`` is a list of (key, label, default, kind) with kind "int",
    "float", "bool", "text" or a tuple of choices.
    """

    def __init__(self, parent, title, fields, note=None):
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.resizable(False, False)
        self.result = None
        self.fields = fields
        self.vars = {}
        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        row = 0
        if note:
            ttk.Label(body, text=note, wraplength=380, foreground="#444444").grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(0, 8))
            row += 1
        first = None
        for key, label, default, kind in fields:
            if kind == "bool":
                var = tk.BooleanVar(value=bool(default))
                widget = ttk.Checkbutton(body, text=label, variable=var)
                widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
            else:
                ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 8))
                var = tk.StringVar(value="" if default is None else str(default))
                if isinstance(kind, tuple):
                    widget = ttk.Combobox(body, textvariable=var, values=kind, state="readonly", width=18)
                else:
                    widget = ttk.Entry(body, textvariable=var, width=20)
                widget.grid(row=row, column=1, sticky="ew", pady=2)
            first = first or widget
            self.vars[key] = var
            row += 1
        buttons = ttk.Frame(body)
        buttons.grid(row=row, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="OK", command=self._ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side=tk.LEFT)
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self.destroy())
        if first is not None:
            first.focus_set()
        self.grab_set()
        self.wait_visibility()

    def _ok(self):
        out = {}
        for key, label, default, kind in self.fields:
            value = self.vars[key].get()
            try:
                if kind == "int":
                    value = int(str(value).strip())
                elif kind == "float":
                    value = float(str(value).strip().replace(",", "."))
            except ValueError:
                messagebox.showerror("Invalid value", f"{label}: {value!r} is not a number.", parent=self)
                return
            out[key] = value
        self.result = out
        self.destroy()


def ask_form(parent, title, fields, note=None):
    """Show a FormDialog and return its result (dict) or None."""
    dialog = FormDialog(parent, title, fields, note)
    parent.wait_window(dialog)
    return dialog.result


class ExportDialog(tk.Toplevel):
    """Choose an output base name and what to write."""

    def __init__(self, parent, default_base, hide_coordinates):
        super().__init__(parent)
        self.title("Export")
        self.transient(parent)
        self.resizable(False, False)
        self.result = None
        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text="Output name (files get suffixes)").grid(row=0, column=0, columnspan=3, sticky="w")
        self.base_var = tk.StringVar(value=default_base)
        ttk.Entry(body, textvariable=self.base_var, width=48).grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(body, text="Browse…", command=self._browse).grid(row=1, column=2, padx=4)
        self.vars = {}
        items = (("csv", "Tree table as CSV  (<name>.csv)", True),
                 ("rds", "Tree table as RDS  (<name>.rds)", True),
                 ("truth", "Ground truth  (<name>-truth.csv)", True),
                 ("recipe", "Recipe copy with seed  (<name>-recipe.json)", True),
                 ("report", "Realism report  (<name>-report.txt)", True),
                 ("image", "Map image", True))
        for i, (key, text, default) in enumerate(items):
            var = tk.BooleanVar(value=default)
            ttk.Checkbutton(body, text=text, variable=var).grid(row=2 + i, column=0, columnspan=2, sticky="w")
            self.vars[key] = var
        self.image_format = tk.StringVar(value="png")
        ttk.Combobox(body, textvariable=self.image_format, values=("png", "svg", "pdf"),
                     state="readonly", width=6).grid(row=7, column=2, sticky="w")
        note = ("Tables are written at full density (a draft is regenerated first).\n"
                "The image uses the current colours and zoom" +
                (" and has no coordinates (hidden)." if hide_coordinates else "."))
        ttk.Label(body, text=note, foreground="#444444").grid(row=8, column=0, columnspan=3, sticky="w", pady=6)
        buttons = ttk.Frame(body)
        buttons.grid(row=9, column=0, columnspan=3, sticky="e")
        ttk.Button(buttons, text="Export", command=self._ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side=tk.LEFT)
        self.bind("<Escape>", lambda e: self.destroy())
        self.grab_set()
        self.wait_visibility()

    def _browse(self):
        path = filedialog.asksaveasfilename(parent=self, title="Output name",
                                            initialfile=os.path.basename(self.base_var.get()))
        if path:
            self.base_var.set(os.path.splitext(path)[0])

    def _ok(self):
        base = self.base_var.get().strip()
        if not base:
            messagebox.showerror("Export", "Give an output name.", parent=self)
            return
        folder = os.path.dirname(os.path.abspath(base))
        if not os.path.isdir(folder):
            messagebox.showerror("Export", f"The folder {folder} does not exist.", parent=self)
            return
        chosen = {k: v.get() for k, v in self.vars.items()}
        if not any(chosen.values()):
            messagebox.showerror("Export", "Choose at least one thing to write.", parent=self)
            return
        self.result = dict(chosen, base=os.path.splitext(base)[0] if base.lower().endswith(
            (".csv", ".rds")) else base, image_format=self.image_format.get())
        self.destroy()


def ask_export(parent, default_base, hide_coordinates):
    dialog = ExportDialog(parent, default_base, hide_coordinates)
    parent.wait_window(dialog)
    return dialog.result


class SettingsDialog(tk.Toplevel):
    """Recipe-wide settings: name, window shape and size, laser artefacts, background.

    With ``hide_coordinates`` the window's centre and vertices are not shown
    and are kept as they are; only sizes can be changed.
    """

    def __init__(self, parent, recipe, on_apply, hide_coordinates=False):
        super().__init__(parent)
        self.title("Settings")
        self.transient(parent)
        self.resizable(False, False)
        self.on_apply = on_apply
        window = recipe["window"]
        self.original_window = window
        self.hidden = hide_coordinates
        if "center" in window:
            self.hidden_centre = list(window["center"])
        else:
            c = shapely.geometry.Polygon(window["vertices"]).centroid
            self.hidden_centre = [round(c.x, 2), round(c.y, 2)]
        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        self.name_var = tk.StringVar(value=recipe["name"])
        ttk.Label(body, text="Recipe name").grid(row=0, column=0, sticky="w")
        ttk.Entry(body, textvariable=self.name_var, width=30).grid(row=0, column=1, columnspan=3, sticky="ew")

        ttk.Label(body, text="Window", font=fonts.get("head")).grid(
            row=1, column=0, sticky="w", pady=(10, 2))
        self.kind_var = tk.StringVar(value=window["kind"])
        kind = ttk.Combobox(body, textvariable=self.kind_var, values=("circle", "rectangle", "polygon"),
                            state="readonly", width=12)
        kind.grid(row=1, column=1, sticky="w", pady=(10, 2))
        kind.bind("<<ComboboxSelected>>", lambda e: self._show_kind())
        centre = window.get("center", self.hidden_centre)
        shown = ("hidden", "hidden") if self.hidden else (f"{centre[0]:.2f}", f"{centre[1]:.2f}")
        self.values = {
            "cx": tk.StringVar(value=shown[0]), "cy": tk.StringVar(value=shown[1]),
            "radius": tk.StringVar(value=f"{window.get('radius', 306.0):g}"),
            "width": tk.StringVar(value=f"{window.get('width', 600.0):g}"),
            "height": tk.StringVar(value=f"{window.get('height', 600.0):g}"),
        }
        self.frames = {}
        for key, rows in (("circle", (("Centre x (m)", "cx"), ("Centre y (m)", "cy"), ("Radius (m)", "radius"))),
                          ("rectangle", (("Centre x (m)", "cx"), ("Centre y (m)", "cy"),
                                         ("Width (m)", "width"), ("Height (m)", "height")))):
            frame = ttk.Frame(body)
            for i, (label, var) in enumerate(rows):
                ttk.Label(frame, text=label).grid(row=i, column=0, sticky="w")
                locked = self.hidden and var in ("cx", "cy")
                ttk.Entry(frame, textvariable=self.values[var], width=14,
                          state="disabled" if locked else "normal").grid(row=i, column=1, sticky="w")
            self.frames[key] = frame
        frame = ttk.Frame(body)
        self.vertices = tk.Text(frame, width=30, height=8)
        if self.hidden:
            ttk.Label(frame, wraplength=320, foreground="#444444", text=(
                "Coordinates are hidden, so the vertices are not shown. An existing polygon "
                "window is kept; untick Hide coordinates to type new vertices.")).pack(anchor="w")
        else:
            ttk.Label(frame, text="Vertices, one \"x y\" pair per line (m):").pack(anchor="w")
            self.vertices.pack(fill=tk.X)
            default_vertices = window.get("vertices") or [[-300, -300], [300, -300], [300, 300], [-300, 300]]
            self.vertices.insert("1.0", "\n".join(f"{x:.2f} {y:.2f}" for x, y in default_vertices))
        self.frames["polygon"] = frame
        if self.hidden:
            ttk.Label(body, text="Coordinates are hidden: the window keeps its centre.",
                      foreground="#444444").grid(row=6, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self._frame_row = 2

        self.laser_var = tk.BooleanVar(value=recipe["laser_artefacts"])
        ttk.Checkbutton(body, text="Laser artefacts (ITD / Simulated, 0.5 m lattice, NA crowns)",
                        variable=self.laser_var).grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))
        self.background_var = tk.BooleanVar(value=recipe["background"]["enabled"])
        ttk.Checkbutton(body, text="Background forest (edit it in the Layers tab)",
                        variable=self.background_var).grid(row=4, column=0, columnspan=4, sticky="w")
        buttons = ttk.Frame(body)
        buttons.grid(row=5, column=0, columnspan=4, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="Apply", command=self._apply).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Close", command=self.destroy).pack(side=tk.LEFT)
        self.body = body
        self._show_kind()
        self.bind("<Escape>", lambda e: self.destroy())
        self.grab_set()

    def _show_kind(self):
        for frame in self.frames.values():
            frame.grid_forget()
        self.frames[self.kind_var.get()].grid(row=self._frame_row, column=0, columnspan=4, sticky="w")

    def _number(self, key):
        if self.hidden and key in ("cx", "cy"):
            return float(self.hidden_centre[0 if key == "cx" else 1])
        return float(self.values[key].get().strip().replace(",", "."))

    def _window(self):
        kind = self.kind_var.get()
        if kind == "circle":
            return {"kind": "circle", "center": [self._number("cx"), self._number("cy")],
                    "radius": self._number("radius")}
        if kind == "rectangle":
            return {"kind": "rectangle", "center": [self._number("cx"), self._number("cy")],
                    "width": self._number("width"), "height": self._number("height")}
        if self.hidden:
            if self.original_window["kind"] != "polygon":
                raise ValueError("untick Hide coordinates to enter the vertices of a polygon")
            return dict(self.original_window)
        vertices = []
        for line in self.vertices.get("1.0", tk.END).splitlines():
            parts = line.replace(",", " ").split()
            if parts:
                if len(parts) != 2:
                    raise ValueError(f"vertex line {line!r} needs two numbers")
                vertices.append([float(parts[0]), float(parts[1])])
        return {"kind": "polygon", "vertices": vertices}

    def _apply(self):
        try:
            window = self._window()
        except ValueError as exc:
            messagebox.showerror("Settings", f"Window: {exc}", parent=self)
            return
        changes = {"name": self.name_var.get().strip(), "window": window,
                   "laser_artefacts": self.laser_var.get(), "background": self.background_var.get()}
        if self.on_apply(changes):
            self.destroy()
