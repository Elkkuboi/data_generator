"""Statistics tabs: report numbers, charts, and G/F/J pattern diagnostics.

The panel adds three tabs to a ttk.Notebook.  ``update`` hands it the
visible trees and the region they come from; only the tab on screen is
recomputed, the others when they are opened.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from engine.patterns import pattern_functions
from engine.report import forest_report, format_report

from . import fonts
from .figures import SPECIES_COLOURS, SPECIES_NAMES

MIN_TREES_FOR_PATTERNS = 10


class StatsPanel:
    """Report, Charts and Patterns tabs."""

    def __init__(self, notebook):
        self.notebook = notebook
        self.data = None
        self._dirty = set()

        self.report_tab = ttk.Frame(notebook)
        self.report_text = tk.Text(self.report_tab, wrap="none", font=fonts.get("fixed"),
                                   height=30, width=58, borderwidth=0)
        scroll = ttk.Scrollbar(self.report_tab, command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=scroll.set, state="disabled")
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.report_text.pack(fill=tk.BOTH, expand=True)

        self.charts_tab, self.charts_fig, self.charts_canvas = self._figure_tab((4.6, 6.2))
        self.patterns_tab, self.patterns_fig, self.patterns_canvas = self._figure_tab((4.6, 6.2))

        notebook.add(self.report_tab, text="Report")
        notebook.add(self.charts_tab, text="Charts")
        notebook.add(self.patterns_tab, text="Patterns")
        notebook.bind("<<NotebookTabChanged>>", lambda e: self.refresh())

    def _figure_tab(self, size):
        frame = ttk.Frame(self.notebook)
        fig = Figure(figsize=size, dpi=90)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return frame, fig, canvas

    def update(self, trees, region, note, scope_text):
        """New data: ``trees`` (visible trees), ``region`` (their area)."""
        self.data = (trees, region, note, scope_text)
        self._dirty = {"report", "charts", "patterns"}
        self.refresh()

    def clear(self):
        self.data = None
        self._dirty = {"report", "charts", "patterns"}
        self.refresh()

    def refresh(self):
        """Recompute the tab that is on screen if its data changed."""
        try:
            current = self.notebook.nametowidget(self.notebook.select())
        except (tk.TclError, KeyError):
            return
        for name, tab, draw in (("report", self.report_tab, self._draw_report),
                                ("charts", self.charts_tab, self._draw_charts),
                                ("patterns", self.patterns_tab, self._draw_patterns)):
            if current is tab and name in self._dirty:
                self._dirty.discard(name)
                draw()

    # ------------------------------------------------------------------

    def _set_report_text(self, text):
        self.report_text.configure(state="normal")
        self.report_text.delete("1.0", tk.END)
        self.report_text.insert("1.0", text)
        self.report_text.configure(state="disabled")

    def _draw_report(self):
        if self.data is None:
            self._set_report_text("No trees yet.")
            return
        trees, region, note, scope = self.data
        if region.area <= 0:
            self._set_report_text("The visible region has no area.")
            return
        text = format_report(forest_report(trees, region, note), title="Realism report")
        self._set_report_text(f"{scope}\n\n{text}")

    def _draw_charts(self):
        fig = self.charts_fig
        fig.clear()
        if self.data is None or len(self.data[0]) == 0:
            fig.text(0.5, 0.5, "No trees to show", ha="center")
            self.charts_canvas.draw_idle()
            return
        trees = self.data[0]
        axes = fig.subplots(3, 1, gridspec_kw={"height_ratios": [1, 1, 0.8]})
        species = trees["Species"].astype(str).to_numpy()
        dbh = trees["DBH"].to_numpy(float)
        height = trees["H"].to_numpy(float)
        for ax, values, label, top in ((axes[0], dbh, "DBH (cm)", 80.0), (axes[1], height, "Height (m)", 40.0)):
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                ax.text(0.5, 0.5, f"no {label}", ha="center", transform=ax.transAxes)
                continue
            hi = min(np.nanpercentile(finite, 99.5), top * 2)
            bins = np.linspace(0, max(hi, 1.0), 40)
            for code, name in SPECIES_NAMES.items():
                sel = (species == code) & np.isfinite(values)
                if sel.any():
                    ax.hist(values[sel], bins=bins, histtype="step", linewidth=1.6,
                            color=SPECIES_COLOURS[code], label=f"{name} ({sel.sum():,})")
            ax.set_xlabel(label, fontsize=8)
            ax.set_ylabel("trees", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, frameon=False)
        ba = np.pi * (np.nan_to_num(dbh) / 200.0) ** 2
        codes = list(SPECIES_NAMES)
        stems = np.array([(species == c).mean() for c in codes])
        basal = np.array([ba[species == c].sum() for c in codes]) / max(ba.sum(), 1e-12)
        pos = np.arange(len(codes))
        axes[2].bar(pos - 0.2, stems, 0.4, color=[SPECIES_COLOURS[c] for c in codes], label="stems")
        axes[2].bar(pos + 0.2, basal, 0.4, color=[SPECIES_COLOURS[c] for c in codes], alpha=0.5,
                    hatch="//", edgecolor="#333333", label="basal area")
        axes[2].set_xticks(pos, [SPECIES_NAMES[c] for c in codes], fontsize=8)
        axes[2].set_ylabel("share", fontsize=8)
        axes[2].set_ylim(0, 1)
        axes[2].tick_params(labelsize=7)
        axes[2].legend(fontsize=7, frameon=False)
        axes[2].set_title("Species shares: stems (solid) and basal area (hatched)", fontsize=8)
        fig.tight_layout()
        self.charts_canvas.draw_idle()

    def _draw_patterns(self):
        fig = self.patterns_fig
        fig.clear()
        if self.data is None or len(self.data[0]) < MIN_TREES_FOR_PATTERNS or self.data[1].area <= 0:
            fig.text(0.5, 0.5, f"Need at least {MIN_TREES_FOR_PATTERNS} visible trees",
                     ha="center")
            self.patterns_canvas.draw_idle()
            return
        trees, region = self.data[0], self.data[1]
        res = pattern_functions(trees["X"].to_numpy(float), trees["Y"].to_numpy(float), region)
        r = res["r"]
        ax1, ax2 = fig.subplots(2, 1)
        ax1.plot(r, res["G"], color="#2458a6", lw=1.8, label="G(r) nearest neighbour")
        ax1.plot(r, res["F"], color="#c8962e", lw=1.8, label="F(r) empty space")
        ax1.plot(r, res["csr"], color="#555555", lw=1.0, ls="--", label="CSR reference")
        ax1.set_xlabel("r (m)", fontsize=8)
        ax1.set_ylim(0, 1)
        ax1.legend(fontsize=7, frameon=False, loc="lower right")
        ax1.tick_params(labelsize=7)
        ax2.plot(r[1:], res["J"][1:], color="#1f5c3a", lw=1.8, label="J(r) = (1 - G) / (1 - F)")
        ax2.axhline(1.0, color="#555555", lw=1.0, ls="--", label="CSR: J = 1")
        ax2.set_xlabel("r (m)", fontsize=8)
        finite = res["J"][np.isfinite(res["J"])]
        top = max(2.0, float(np.nanpercentile(finite, 95)) * 1.1) if len(finite) else 2.0
        ax2.set_ylim(0, min(top, 20.0))
        ax2.legend(fontsize=7, frameon=False, loc="upper left")
        ax2.tick_params(labelsize=7)
        ax2.text(0.99, 0.02, "J < 1 clustered, J > 1 regular", ha="right", va="bottom",
                 transform=ax2.transAxes, fontsize=7, color="#555555")
        fig.suptitle(f"{res['n']:,} visible trees, {res['area'] / 1e4:.2f} ha\n"
                     "No edge correction: G, F and J are biased near the edges", fontsize=8)
        fig.tight_layout()
        self.patterns_canvas.draw_idle()
