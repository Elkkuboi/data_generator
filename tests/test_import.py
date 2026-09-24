"""Editing imported tree files: a CSV or RDS file as the background of a recipe."""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import freehand_points
from engine.generate import generate, lattice_offset
from engine.geometry import raw_shape, window_geometry
from engine.io import fit_window, infer_lattice_offset, read_trees, recipe_for_file, write_forest
from engine.recipe import RecipeError, new_recipe, validate_recipe

STRINGS = ["Species", "Status", "Type"]


def as_objects(trees):
    return trees.astype({c: object for c in STRINGS}).reset_index(drop=True)


def inside(shape, trees):
    return shapely.intersects_xy(raw_shape(shape), trees["X"].to_numpy(), trees["Y"].to_numpy())


class FileBackgroundTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        recipe = new_recipe(seed=3)
        recipe["layers"] = [{"type": "clustered", "mode": "replace",
                             "shape": {"kind": "circle", "center": [100, 100], "radius": 80}}]
        cls.source = generate(recipe)
        cls.base = os.path.join(cls.tmp, "source")
        write_forest(cls.source, cls.base, ("csv", "rds"))
        cls.csv, cls.rds = cls.base + ".csv", cls.base + ".rds"
        cls.window = cls.source.recipe["window"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def recipe(self, layers=(), path=None, **extra):
        recipe = {"seed": 99, "window": self.window, "background": {"file": path or self.csv},
                  "layers": list(layers)}
        recipe.update(extra)
        return recipe

    def test_no_layers_reproduces_the_file_exactly(self):
        for path in (self.csv, self.rds):
            with self.subTest(file=os.path.basename(path)):
                forest = generate(self.recipe(path=path))
                pd.testing.assert_frame_equal(as_objects(forest.trees), as_objects(read_trees(path)))
                self.assertEqual(set(forest.truth["layer_type"]), {"file"})
                self.assertEqual(set(forest.truth["layer_name"]), {os.path.basename(path)})

    def test_relative_path_and_recorded_absolute_path(self):
        forest = generate(self.recipe(path="source.csv"), base_dir=self.tmp)
        self.assertEqual(len(forest.trees), len(self.source.trees))
        self.assertEqual(forest.recipe["background"]["file"], os.path.abspath(self.csv))
        again = generate(forest.recipe)            # the recipe copy is self-contained
        pd.testing.assert_frame_equal(again.trees, forest.trees)

    def test_clear_removes_imported_trees_only_in_shape(self):
        stroke = {"kind": "corridor", "points": freehand_points(), "width": 8}
        forest = generate(self.recipe([{"type": "clear", "shape": stroke}]))
        original = read_trees(self.csv)
        self.assertFalse(inside(stroke, forest.trees).any())
        kept = original[~inside(stroke, original)]
        pd.testing.assert_frame_equal(as_objects(forest.trees), as_objects(kept))

    def test_thin_replace_and_add(self):
        area = {"kind": "rectangle", "center": [-100, -50], "width": 200, "height": 150}
        original = read_trees(self.csv)
        n_area = int(inside(area, original).sum())

        thinned = generate(self.recipe([{"type": "thin", "params": {"fraction": 0.5}, "shape": area}]))
        ratio = inside(area, thinned.trees).sum() / n_area
        self.assertLess(abs(ratio - 0.5), 0.05)

        replaced = generate(self.recipe([{"id": "P", "preset": "pine_heath", "mode": "replace",
                                          "shape": area}]))
        in_area = inside(area, replaced.trees)
        self.assertTrue((replaced.truth["layer_id"][in_area] == "P").all())
        self.assertGreater(in_area.sum(), 0.9 * 450 * 3.0)

        added = generate(self.recipe([{"id": "U", "type": "random", "mode": "add",
                                       "params": {"density": 300}, "shape": area}]))
        counts = added.truth["layer_id"][inside(area, added.trees)].value_counts()
        self.assertEqual(counts["background"], n_area)
        self.assertGreater(counts["U"], 700)

    def test_imported_rows_keep_their_values(self):
        forest = generate(self.recipe([
            {"type": "clear", "shape": {"kind": "circle", "center": [0, 0], "radius": 100}},
            {"type": "random", "params": {"density": 500}, "shape": {"kind": "window"}}]))
        mine = forest.truth["layer_id"] == "background"
        original = as_objects(read_trees(self.csv)).set_index(["X", "Y"])
        kept = as_objects(forest.trees[mine]).set_index(["X", "Y"])
        pd.testing.assert_frame_equal(kept, original.loc[kept.index])

    def test_new_itd_trees_join_the_files_lattice(self):
        offset = infer_lattice_offset(read_trees(self.csv))
        self.assertIsNotNone(offset)
        self.assertNotEqual(offset, lattice_offset(12345))     # the recipe's own lattice differs
        forest = generate(self.recipe([{"id": "N", "type": "random", "params": {"density": 800},
                                        "composition": {"size_class": "mature"},
                                        "shape": {"kind": "window"}}], seed=12345))
        new = forest.trees[(forest.truth["layer_id"] == "N") & (forest.trees["Type"] == "ITD")]
        self.assertGreater(len(new), 100)
        for column, origin in zip(("X", "Y"), offset):
            steps = (new[column].to_numpy() - origin) / 0.5
            self.assertLess(np.abs(steps - np.round(steps)).max(), 1e-6)

    def test_trees_outside_the_window_are_left_out(self):
        small = {"kind": "circle", "center": [0, 0], "radius": 150}
        forest = generate(self.recipe(window=small))
        self.assertTrue(shapely.contains_xy(window_geometry(small), forest.trees["X"].to_numpy(),
                                            forest.trees["Y"].to_numpy()).all())
        self.assertTrue(any("outside the window" in w for w in forest.warnings))

    def test_draft_uses_a_share_of_the_file(self):
        forest = generate(self.recipe(), density_scale=0.1)
        self.assertLess(abs(len(forest.trees) / len(self.source.trees) - 0.1), 0.01)

    def test_unusual_files_survive_unchanged(self):
        path = os.path.join(self.tmp, "odd.csv")
        pd.DataFrame({"x": [1.0, 2.5, 2.5, 40.0], "y": [1.0, 3.0, 3.0, 20.0],
                      "DBH": [12.0, np.nan, 30.0, 8.0], "Species": ["2", "9", "1", "3"]}).to_csv(path, index=False)
        recipe = {"window": {"kind": "circle", "center": [0, 0], "radius": 100},
                  "background": {"file": path},
                  "layers": [{"type": "thin", "params": {"fraction": 1.0, "dbh_below": 10},
                              "shape": {"kind": "window"}}]}
        trees = generate(recipe).trees
        self.assertEqual(list(trees["Species"]), ["2", "9", "1"])     # 8 cm tree thinned
        self.assertTrue(np.isnan(trees["DBH"].iloc[1]))                # NA dbh never thinned by dbh
        self.assertEqual(len(trees.drop_duplicates(["X", "Y"])), 2)    # duplicates in a file are kept

    def test_errors_name_the_key(self):
        with self.assertRaises(RecipeError) as ctx:
            generate(self.recipe(path=os.path.join(self.tmp, "missing.csv")))
        self.assertIn('background, key "file"', str(ctx.exception))
        with self.assertRaises(RecipeError) as ctx:
            validate_recipe({"background": {"file": self.csv, "type": "random"}})
        self.assertIn('"type"', str(ctx.exception))
        with self.assertRaises(RecipeError):
            validate_recipe({"background": {"file": "trees.xlsx"}})


class FitWindowTest(unittest.TestCase):
    def check_contains(self, x, y):
        spec = fit_window(pd.DataFrame({"X": x, "Y": y}))
        self.assertTrue(shapely.contains_xy(window_geometry(spec), x, y).all())
        return spec

    def test_circular_plot_gives_a_circle(self):
        rng = np.random.default_rng(0)
        r, a = 306 * np.sqrt(rng.random(20000)), rng.random(20000) * 2 * np.pi
        x, y = 385000 + r * np.cos(a), 6700000 + r * np.sin(a)
        spec = self.check_contains(np.round(x, 2), np.round(y, 2))
        self.assertEqual(spec["kind"], "circle")
        self.assertAlmostEqual(spec["radius"], 306.5, delta=1.0)

    def test_other_shapes_give_a_polygon(self):
        rng = np.random.default_rng(1)
        x, y = rng.uniform(0, 400, 5000), rng.uniform(0, 100, 5000)
        self.assertEqual(self.check_contains(x, y)["kind"], "polygon")

    def test_recipe_for_file_uses_the_sidecar_window(self):
        tmp = tempfile.mkdtemp()
        try:
            recipe = new_recipe()
            recipe["window"] = {"kind": "rectangle", "center": [0, 0], "width": 300, "height": 200}
            forest = generate(recipe)
            base = os.path.join(tmp, "plot")
            write_forest(forest, base, ("csv",))
            made = recipe_for_file(base + ".csv", forest.trees)
            self.assertEqual(made["window"], recipe["window"])
            self.assertEqual(made["name"], "plot_edited")
            self.assertEqual(made["background"]["file"], os.path.abspath(base + ".csv"))
            os.remove(base + "-recipe.json")
            self.assertEqual(recipe_for_file(base + ".csv", forest.trees)["window"]["kind"], "polygon")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
