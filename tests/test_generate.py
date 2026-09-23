"""Generation: determinism, windows, densities, layer semantics, defaults."""

import os
import sys
import time
import unittest

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import WINDOWS, background_recipe, freehand_points
from engine.generate import generate, lattice_offset
from engine.geometry import raw_shape, window_geometry
from engine.presets import (
    DEFAULT_DEAD_SHARE,
    DEFAULT_IMPUTED_SHARE,
    DEFAULT_MEDIAN_DBH,
    DEFAULT_MEDIAN_HEIGHT,
    DEFAULT_SHARES,
    SPECIES_CODES,
)
from engine.recipe import new_recipe

MIXED_RECIPE = {
    "name": "mixed",
    "seed": 11,
    "layers": [
        {"id": "A", "preset": "young_spruce", "mode": "replace",
         "shape": {"kind": "circle", "center": [-100, 50], "radius": 80}},
        {"id": "B", "type": "rows", "mode": "replace", "params": {"angle": 20},
         "shape": {"kind": "rectangle", "center": [120, -60], "width": 120, "height": 80}},
        {"id": "C", "type": "regular", "params": {"density": 400, "min_distance": 3},
         "shape": {"kind": "polygon", "vertices": [[0, 150], [150, 150], [100, 260]]}},
        {"id": "D", "type": "gradient", "shape": {"kind": "circle", "center": [0, -200], "radius": 60}},
        {"id": "E", "type": "clear", "shape": {"kind": "corridor", "points": [[-306, 0], [306, 20]], "width": 5}},
        {"id": "F", "type": "thin", "params": {"fraction": 0.5, "dbh_below": 10},
         "shape": {"kind": "circle", "center": [150, 100], "radius": 70}},
    ],
}


def trees_in(geom, trees):
    return shapely.intersects_xy(geom, trees["X"].to_numpy(), trees["Y"].to_numpy())


class DeterminismTest(unittest.TestCase):
    def test_same_seed_identical(self):
        a = generate(MIXED_RECIPE, seed=5)
        b = generate(MIXED_RECIPE, seed=5)
        pd.testing.assert_frame_equal(a.trees, b.trees)
        pd.testing.assert_frame_equal(a.truth, b.truth)

    def test_different_seed_differs(self):
        a = generate(MIXED_RECIPE, seed=5).trees
        b = generate(MIXED_RECIPE, seed=6).trees
        self.assertFalse(len(a) == len(b) and np.allclose(a["X"], b["X"]))

    def test_recipe_seed_used_and_recorded(self):
        forest = generate(MIXED_RECIPE)
        self.assertEqual(forest.recipe["seed"], 11)
        forest = generate(MIXED_RECIPE, seed=99)
        self.assertEqual(forest.recipe["seed"], 99)

    def test_editing_one_layer_keeps_others(self):
        edited = {**MIXED_RECIPE, "layers": [dict(layer) for layer in MIXED_RECIPE["layers"]]}
        edited["layers"][1] = dict(edited["layers"][1], params={"angle": 50})
        a, b = generate(MIXED_RECIPE), generate(edited)
        keep_a = a.trees[a.truth["layer_id"] == "C"].reset_index(drop=True)
        keep_b = b.trees[b.truth["layer_id"] == "C"].reset_index(drop=True)
        pd.testing.assert_frame_equal(keep_a, keep_b)


class WindowTest(unittest.TestCase):
    def test_all_trees_inside_every_window(self):
        for name, window in WINDOWS.items():
            for laser in (True, False):
                recipe = dict(MIXED_RECIPE, window=window, laser_artefacts=laser)
                forest = generate(recipe)
                geom = window_geometry(window)
                inside = shapely.contains_xy(geom, forest.trees["X"].to_numpy(), forest.trees["Y"].to_numpy())
                with self.subTest(window=name, laser=laser):
                    self.assertGreater(len(forest.trees), 100)
                    self.assertTrue(inside.all())

    def test_default_window(self):
        forest = generate(new_recipe())
        self.assertAlmostEqual(forest.window.area / 1e4, np.pi * 306 ** 2 / 1e4, delta=0.05)
        r = np.hypot(forest.trees["X"], forest.trees["Y"])
        self.assertLess(r.max(), 306.0)


class DensityTest(unittest.TestCase):
    def check_density(self, kind, params, expected, tol=0.04):
        forest = generate(background_recipe(kind, params, laser=True))
        realised = len(forest.trees) / (forest.window.area / 1e4)
        self.assertLess(abs(realised / expected - 1), tol, f"{kind}: {realised:.1f} vs {expected}")

    def test_random(self):
        self.check_density("random", {"density": 650}, 650)
        self.check_density("random", {"density": 120}, 120, tol=0.06)

    def test_clustered(self):
        self.check_density("clustered", {"cluster_density": 50, "trees_per_cluster": 13,
                                         "cluster_radius": 4}, 650, tol=0.06)

    def test_regular(self):
        self.check_density("regular", {"density": 650, "min_distance": 2.5}, 650)

    def test_rows(self):
        self.check_density("rows", {"row_spacing": 4, "tree_spacing": 2.5, "angle": 33,
                                    "jitter": 0.3}, 1000)

    def test_gradient(self):
        self.check_density("gradient", {"density_start": 200, "density_end": 1000,
                                        "direction": 45}, 600, tol=0.06)

    def test_draft_scale(self):
        full = generate(new_recipe())
        draft = generate(new_recipe(), density_scale=0.1)
        self.assertLess(abs(len(draft.trees) / len(full.trees) - 0.1), 0.01)

    def test_unreachable_regular_density_reports_achieved(self):
        forest = generate(background_recipe("regular", {"density": 1500, "min_distance": 2.5},
                                            window=WINDOWS["rectangle"]))
        self.assertTrue(any("reached" in w for w in forest.warnings), forest.warnings)
        self.assertLess(len(forest.trees) / (forest.window.area / 1e4), 1500)


class RegularMinimumDistanceTest(unittest.TestCase):
    def test_never_violated(self):
        from scipy.spatial import cKDTree
        for laser in (True, False):
            for density, r in ((650, 2.5), (1000, 2.5), (300, 4.0)):
                forest = generate(background_recipe("regular", {"density": density, "min_distance": r},
                                                    laser=laser, window=WINDOWS["circle"]))
                xy = forest.trees[["X", "Y"]].to_numpy()
                d, _ = cKDTree(xy).query(xy, k=2)
                with self.subTest(laser=laser, density=density, r=r):
                    self.assertGreaterEqual(d[:, 1].min(), r - 1e-9)


class LayerSemanticsTest(unittest.TestCase):
    SHAPES = {
        "window": {"kind": "window"},
        "circle": {"kind": "circle", "center": [30, 40], "radius": 60},
        "rectangle": {"kind": "rectangle", "center": [-50, -40], "width": 100, "height": 60},
        "polygon": {"kind": "polygon", "vertices": [[0, 0], [150, -20], [120, 120], [40, 60]]},
        "corridor": {"kind": "corridor", "points": [[-306, -100], [306, 100]], "width": 6},
        "freehand": {"kind": "corridor", "points": freehand_points(), "width": 7.5},
    }

    def test_clear_leaves_no_trees_in_shape(self):
        for name, shape in self.SHAPES.items():
            recipe = new_recipe()
            recipe["layers"] = [{"type": "clear", "shape": shape}]
            forest = generate(recipe)
            with self.subTest(shape=name):
                self.assertFalse(trees_in(raw_shape(shape) if name != "window"
                                          else forest.window, forest.trees).any())
                if name != "window":
                    self.assertGreater(len(forest.trees), 1000)

    def test_later_add_layer_can_fill_cleared_area(self):
        recipe = new_recipe()
        shape = self.SHAPES["circle"]
        recipe["layers"] = [{"type": "clear", "shape": shape},
                            {"type": "random", "params": {"density": 300}, "shape": shape}]
        forest = generate(recipe)
        inside = trees_in(raw_shape(shape), forest.trees)
        self.assertTrue((forest.truth["layer_id"][inside] == "L2").all())

    def test_replace_versus_add(self):
        shape = self.SHAPES["circle"]
        geom = raw_shape(shape)
        counts = {}
        for mode in ("add", "replace"):
            recipe = new_recipe()
            recipe["layers"] = [{"id": "P", "type": "random", "mode": mode,
                                 "params": {"density": 1000}, "shape": shape}]
            forest = generate(recipe)
            inside = trees_in(geom, forest.trees)
            counts[mode] = forest.truth["layer_id"][inside].value_counts().to_dict()
        self.assertGreater(counts["add"].get("background", 0), 300)
        self.assertEqual(counts["replace"].get("background", 0), 0)
        self.assertGreater(counts["replace"]["P"], 800)

    def test_thin_fraction_and_dbh_filter(self):
        shape = {"kind": "rectangle", "center": [0, 0], "width": 300, "height": 200}
        base = new_recipe()
        before = generate(base).trees
        base["layers"] = [{"type": "thin", "params": {"fraction": 0.5, "dbh_below": 10},
                           "shape": shape}]
        after = generate(base).trees
        geom = raw_shape(shape)
        small = lambda t: trees_in(geom, t) & (t["DBH"] < 10).to_numpy()
        large = lambda t: trees_in(geom, t) & (t["DBH"] >= 10).to_numpy()
        self.assertEqual(large(before).sum(), large(after).sum())
        ratio = small(after).sum() / small(before).sum()
        self.assertLess(abs(ratio - 0.5), 0.05)

    def test_layer_outside_window_warns(self):
        recipe = new_recipe()
        recipe["layers"] = [{"name": "far", "type": "random",
                             "shape": {"kind": "circle", "center": [5000, 0], "radius": 10}}]
        forest = generate(recipe)
        self.assertTrue(any("outside" in w for w in forest.warnings))

    def test_disabled_layers_are_ignored(self):
        recipe = new_recipe()
        recipe["layers"] = [{"type": "clear", "enabled": False, "shape": {"kind": "window"}}]
        self.assertGreater(len(generate(recipe).trees), 10_000)
        recipe["background"] = None
        self.assertEqual(len(generate(recipe).trees), 0)

    def test_truth_matches_layers(self):
        forest = generate(MIXED_RECIPE)
        self.assertEqual(len(forest.truth), len(forest.trees))
        self.assertEqual(list(forest.truth.columns), ["layer_id", "layer_name", "layer_type", "preset"])
        self.assertEqual(set(forest.truth["layer_id"]), {"background", "A", "B", "C", "D"})
        self.assertTrue((forest.truth["preset"][forest.truth["layer_id"] == "A"] == "young_spruce").all())
        self.assertEqual(forest.layer_counts["E"], 0)


class CompositionTest(unittest.TestCase):
    def test_species_and_dead_shares(self):
        comp = {"species": {"pine": 0.6, "spruce": 0.1, "broadleaved": 0.3}, "dead_share": 0.2}
        trees = generate(background_recipe("random", {"density": 800}, composition=comp)).trees
        shares = trees["Species"].value_counts(normalize=True)
        for code, share in zip(SPECIES_CODES, (0.6, 0.1, 0.3)):
            self.assertAlmostEqual(shares[code], share, delta=0.015)
        self.assertAlmostEqual((trees["Status"] == "Dead").mean(), 0.2, delta=0.015)

    def test_size_class_median(self):
        comp = {"size_class": "mature"}
        trees = generate(background_recipe("random", {"density": 300}, composition=comp)).trees
        self.assertAlmostEqual(trees["DBH"].median(), 26.0, delta=0.8)

    def test_dbh_range(self):
        comp = {"dbh": {"median": 60, "sigma": 1.0}}
        trees = generate(background_recipe("random", {"density": 300}, composition=comp)).trees
        self.assertGreaterEqual(trees["DBH"].min(), 4.5)
        self.assertLessEqual(trees["DBH"].max(), 150.0)
        self.assertTrue((trees["Latvus_h"].dropna() <= trees["H"][trees["Latvus_h"].notna()]).all())


class DefaultsTest(unittest.TestCase):
    """A recipe with no layers reproduces the defaults table."""

    @classmethod
    def setUpClass(cls):
        cls.forest = generate(new_recipe(), seed=2024)
        cls.trees = cls.forest.trees

    def test_density(self):
        density = len(self.trees) / (self.forest.window.area / 1e4)
        self.assertLess(abs(density / 650 - 1), 0.03)

    def test_species_shares(self):
        shares = self.trees["Species"].value_counts(normalize=True)
        for code, share in zip(SPECIES_CODES, DEFAULT_SHARES):
            self.assertAlmostEqual(shares[code], share, delta=0.015)

    def test_medians_per_species(self):
        for i, code in enumerate(SPECIES_CODES):
            sub = self.trees[self.trees["Species"] == code]
            with self.subTest(species=code):
                self.assertAlmostEqual(sub["DBH"].median(), DEFAULT_MEDIAN_DBH[i], delta=0.6)
                self.assertAlmostEqual(sub["H"].median(), DEFAULT_MEDIAN_HEIGHT[i], delta=0.5)

    def test_dead_shares(self):
        for i, code in enumerate(SPECIES_CODES):
            sub = self.trees[self.trees["Species"] == code]
            dead = (sub["Status"] == "Dead").mean()
            self.assertAlmostEqual(dead, DEFAULT_DEAD_SHARE[i], delta=0.012)

    def test_imputed_share(self):
        self.assertAlmostEqual((self.trees["Type"] == "Simulated").mean(),
                               DEFAULT_IMPUTED_SHARE, delta=0.04)

    def test_allometry(self):
        itd = self.trees[self.trees["Type"] == "ITD"]
        ff = {"1": 0.51, "2": 0.52, "3": 0.48}
        for code, k in (("1", 0.11), ("2", 0.12), ("3", 0.12)):
            sub = itd[itd["Species"] == code]
            self.assertAlmostEqual((sub["Latvus_d"] / sub["DBH"]).median(), k, delta=0.01)
            v = ff[code] * np.pi * (sub["DBH"] / 200) ** 2 * sub["H"]
            self.assertLess(np.abs(v - sub["V"]).max(), 1e-4 + 1e-12)
        ratio = itd["Latvus_h"] / itd["H"]
        self.assertTrue((ratio <= 1.0).all())
        for code, target in (("1", 0.54), ("2", 0.69), ("3", 0.58)):
            self.assertAlmostEqual(ratio[itd["Species"] == code].median(), target, delta=0.03)


class LaserArtefactTest(unittest.TestCase):
    def test_laser_on(self):
        forest = generate(dict(MIXED_RECIPE, laser_artefacts=True))
        trees = forest.trees
        itd = trees["Type"] == "ITD"
        ox, oy = lattice_offset(forest.recipe["seed"])
        for column, offset in (("X", ox), ("Y", oy)):
            steps = (trees.loc[itd, column] - offset) / 0.5
            self.assertLess(np.abs(steps - np.round(steps)).max(), 1e-6)
        crowns_missing = trees["Latvus_h"].isna() | trees["Latvus_d"].isna()
        self.assertTrue((crowns_missing == ~itd).all())
        self.assertTrue(set(trees["Type"]) == {"ITD", "Simulated"})

    def test_itd_share_follows_dbh(self):
        trees = generate(new_recipe()).trees
        small = trees[trees["DBH"] < 12]
        big = trees[trees["DBH"] > 30]
        self.assertLess((small["Type"] == "ITD").mean(), 0.15)
        self.assertGreater((big["Type"] == "ITD").mean(), 0.95)

    def test_laser_off(self):
        trees = generate(dict(MIXED_RECIPE, laser_artefacts=False)).trees
        self.assertTrue((trees["Type"] == "ITD").all())
        self.assertFalse(trees[["Latvus_h", "Latvus_d"]].isna().any().any())
        on_lattice = np.abs(trees["X"] / 0.5 - np.round(trees["X"] / 0.5)) < 1e-6
        self.assertLess(on_lattice.mean(), 0.1)

    def test_no_duplicate_positions(self):
        trees = generate(new_recipe()).trees
        self.assertFalse(trees.duplicated(["X", "Y"]).any())


class PerformanceTest(unittest.TestCase):
    def test_default_is_fast(self):
        start = time.perf_counter()
        generate(new_recipe())
        self.assertLess(time.perf_counter() - start, 2.0)

    def test_45000_trees(self):
        recipe = background_recipe("clustered", {"cluster_density": 100, "trees_per_cluster": 15.3,
                                                 "cluster_radius": 5}, laser=True)
        recipe["layers"] = [
            {"type": "random", "mode": "replace", "params": {"density": 2000},
             "shape": {"kind": "circle", "center": [100, 0], "radius": 80}},
            {"type": "clear", "shape": {"kind": "corridor", "points": freehand_points(), "width": 5}},
        ]
        start = time.perf_counter()
        forest = generate(recipe)
        elapsed = time.perf_counter() - start
        self.assertGreater(len(forest.trees), 42_000)
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
