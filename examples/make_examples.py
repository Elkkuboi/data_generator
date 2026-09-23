"""Rebuild the example recipes and their PNG maps.

    python examples/make_examples.py

The recipes are ordinary JSON files; this script only records how they were
made (the generators are seeded, so the output is always the same).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from engine.generate import generate  # noqa: E402
from engine.generators import apply_generator  # noqa: E402
from engine.geometry import shape_geometry  # noqa: E402
from engine.recipe import new_recipe, save_recipe, validate_recipe  # noqa: E402
from gui.figures import layer_colour, save_map  # noqa: E402


def default():
    recipe = new_recipe("default", seed=42)
    recipe["description"] = ("The default forest: no layers, only the background "
                             "(about 650 trees/ha, default species mix and sizes).")
    return recipe


def mosaic_with_trail():
    recipe = new_recipe("mosaic_with_trail", seed=42)
    recipe["description"] = ("Eight Voronoi stand patches, each with a random preset, "
                             "crossed by a 5 m wide trail.")
    recipe = apply_generator(recipe, "mosaic", np.random.default_rng(7), n_patches=8)
    recipe = apply_generator(recipe, "trails", np.random.default_rng(3), n_trails=1, width=5.0)
    return recipe


def clearing_and_rows():
    recipe = new_recipe("clearing_and_rows", seed=42)
    recipe["description"] = ("A planted pine stand in rows, a clearing with regeneration "
                             "along its edge, a thinned area and a brushed trail.")
    ring = [[round(-80 + 40 * np.cos(a), 2), round(60 + 40 * np.sin(a), 2)]
            for a in np.linspace(0, 2 * np.pi, 49)]
    recipe["layers"] = [
        {"id": "L1", "name": "Planted pine", "type": "rows", "mode": "replace",
         "preset": "managed_rows", "params": {"angle": 30.0},
         "shape": {"kind": "rectangle", "center": [120.0, -40.0], "width": 150.0, "height": 90.0}},
        {"id": "L2", "name": "Clearing", "type": "clear",
         "shape": {"kind": "circle", "center": [-80.0, 60.0], "radius": 40.0}},
        {"id": "L3", "name": "Clearing edge regeneration", "type": "random", "mode": "add",
         "preset": "seedling_stand", "params": {"density": 1500.0},
         "shape": {"kind": "corridor", "points": ring, "width": 12.0}},
        {"id": "L4", "name": "Thinned from below", "type": "thin",
         "params": {"fraction": 0.6, "dbh_below": 12.0},
         "shape": {"kind": "polygon",
                   "vertices": [[-220.0, -100.0], [-50.0, -170.0], [-20.0, -20.0], [-190.0, 10.0]]}},
        {"id": "L5", "name": "Trail (brush stroke)", "type": "clear",
         "shape": {"kind": "corridor", "width": 4.0,
                   "points": [[-310.0, -10.0], [-200.0, 12.0], [-100.0, 20.0], [0.0, -5.0],
                              [100.0, 25.0], [200.0, 40.0], [310.0, 30.0]]}},
    ]
    return validate_recipe(recipe)


def old_growth_patch():
    recipe = new_recipe("old_growth_patch", seed=42)
    recipe["description"] = ("An old-growth patch with understorey spruce and two bark-beetle "
                             "spots inside an ordinary managed forest.")
    patch = {"kind": "polygon", "vertices": [[-150.0, -40.0], [-40.0, -140.0], [90.0, -110.0],
                                             [140.0, 20.0], [60.0, 130.0], [-90.0, 110.0]]}
    beetle = {"type": "clustered", "mode": "replace",
              "params": {"cluster_density": 40.0, "trees_per_cluster": 12.0, "cluster_radius": 4.0},
              "composition": {"species": {"pine": 0.05, "spruce": 0.9, "broadleaved": 0.05},
                              "size_class": "mature", "dead_share": 0.7}}
    recipe["layers"] = [
        {"id": "L1", "name": "Old growth", "preset": "old_growth", "mode": "replace",
         "type": "clustered", "shape": patch},
        {"id": "L2", "name": "Understorey", "type": "clustered", "mode": "add",
         "params": {"cluster_density": 25.0, "trees_per_cluster": 10.0, "cluster_radius": 4.0},
         "composition": {"species": {"pine": 0.05, "spruce": 0.75, "broadleaved": 0.2},
                         "size_class": "young", "dead_share": 0.01},
         "shape": patch},
        dict(beetle, id="L3", name="Beetle spot 1",
             shape={"kind": "circle", "center": [20.0, 40.0], "radius": 18.0}),
        dict(beetle, id="L4", name="Beetle spot 2",
             shape={"kind": "circle", "center": [45.0, 10.0], "radius": 12.0}),
    ]
    return validate_recipe(recipe)


EXAMPLES = (default, mosaic_with_trail, clearing_and_rows, old_growth_patch)


def main():
    for build in EXAMPLES:
        recipe = build()
        name = recipe["name"]
        save_recipe(recipe, os.path.join(HERE, name + ".json"))
        forest = generate(recipe)
        colour_by = "status" if name == "old_growth_patch" else "species"
        overlays = [(shape_geometry(layer["shape"], forest.window), layer_colour(i + 1))
                    for i, layer in enumerate(recipe["layers"])]
        save_map(os.path.join(HERE, name + ".png"), forest.trees, forest.window,
                 colour_by=colour_by, truth=forest.truth, overlays=overlays,
                 title=f"{name}: {len(forest.trees):,} trees, seed {recipe['seed']}",
                 size_in=6.0, dpi=100)
        print(f"{name}: {len(forest.trees):,} trees")


if __name__ == "__main__":
    main()
