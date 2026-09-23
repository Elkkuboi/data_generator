"""Shared helpers for the engine tests."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

EXAMPLES_DIR = os.path.join(ROOT, "examples")

WINDOWS = {
    "circle": {"kind": "circle", "center": [10.0, -20.0], "radius": 150.0},
    "rectangle": {"kind": "rectangle", "center": [0.0, 0.0], "width": 300.0, "height": 180.0},
    "polygon": {"kind": "polygon",
                "vertices": [[0, 0], [250, 0], [250, 120], [120, 60], [0, 200]]},
}


def background_recipe(kind, params=None, laser=False, seed=7, window=None, composition=None):
    """A recipe whose whole forest is one background layer of ``kind``."""
    background = {"type": kind, "params": params or {}}
    if composition:
        background["composition"] = composition
    recipe = {"name": f"test_{kind}", "seed": seed, "laser_artefacts": laser,
              "background": background}
    if window:
        recipe["window"] = window
    return recipe


def freehand_points(n=60, seed=0):
    """A wiggly polyline like a brush stroke."""
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.linspace(-200, 200, n)
    return [[float(x), float(40 * np.sin(x / 30) + rng.normal(0, 2))] for x in t]
