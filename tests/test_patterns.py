"""Pattern diagnostics: Clark-Evans and J(r) separate clustered, random and regular."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import background_recipe
from engine.generate import generate
from engine.patterns import clark_evans, pattern_functions
from engine.report import forest_report, format_report

PATTERNS = {
    "clustered": ("clustered", {"cluster_density": 40, "trees_per_cluster": 16, "cluster_radius": 5}),
    "random": ("random", {"density": 650}),
    "regular": ("regular", {"density": 650, "min_distance": 2.5}),
}


class PatternTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for name, (kind, params) in PATTERNS.items():
            for laser in (False, True):
                forest = generate(background_recipe(kind, params, laser=laser, seed=3))
                x, y = forest.trees["X"].to_numpy(), forest.trees["Y"].to_numpy()
                ce = clark_evans(x, y, forest.window.area, forest.window.length)
                funcs = pattern_functions(x, y, forest.window)
                lam = len(x) / forest.window.area
                k = int(np.argmin(np.abs(funcs["r"] - 0.5 / np.sqrt(lam))))
                cls.results[name, laser] = (ce, funcs["J"][k], funcs)

    def test_clark_evans(self):
        for laser in (False, True):
            self.assertLess(self.results["clustered", laser][0], 0.9)
            self.assertAlmostEqual(self.results["random", laser][0], 1.0, delta=0.05)
            self.assertGreater(self.results["regular", laser][0], 1.2)

    def test_j_function(self):
        for laser in (False, True):
            self.assertLess(self.results["clustered", laser][1], 0.85)
            self.assertAlmostEqual(self.results["random", laser][1], 1.0, delta=0.1)
            self.assertGreater(self.results["regular", laser][1], 1.3)

    def test_random_g_and_f_follow_csr(self):
        funcs = self.results["random", False][2]
        self.assertLess(np.nanmax(np.abs(funcs["G"] - funcs["csr"])), 0.03)
        self.assertLess(np.nanmax(np.abs(funcs["F"] - funcs["csr"])), 0.03)


class ReportTest(unittest.TestCase):
    def test_report_contains_targets(self):
        forest = generate(background_recipe("random", {"density": 650}, laser=True))
        rep = forest_report(forest.trees, forest.window, "test window")
        self.assertEqual(rep["n"], len(forest.trees))
        self.assertAlmostEqual(sum(s["share_stems"] for s in rep["species"].values()), 1.0)
        self.assertAlmostEqual(sum(s["share_ba"] for s in rep["species"].values()), 1.0)
        text = format_report(rep)
        for phrase in ("Density (trees/ha)", "Basal area", "Volume", "Clark-Evans", "Imputed share",
                       "share by basal area", "dbh quantiles", "height quantiles", "650"):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
