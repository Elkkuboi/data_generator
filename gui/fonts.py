"""Named Tk fonts derived from the platform defaults (created on first use)."""

from tkinter import font as tkfont

_cache = {}


def get(kind):
    """Return a named font: "bold", "head", "small" or "fixed"."""
    if kind not in _cache:
        if kind == "fixed":
            _cache[kind] = tkfont.nametofont("TkFixedFont")
        else:
            base = tkfont.nametofont("TkDefaultFont")
            f = base.copy()
            size = base.cget("size")
            if kind == "bold":
                f.configure(weight="bold")
            elif kind == "head":
                f.configure(weight="bold", size=size + (1 if size > 0 else -1))
            elif kind == "small":
                f.configure(size=size - 1 if size > 0 else size + 1)
            _cache[kind] = f
    return _cache[kind]
