"""Output format: columns, dtypes, CSV text, RDS round trip (and R, if installed)."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import ROOT
from engine.generate import COLUMNS, generate
from engine.io import read_trees, write_csv, write_forest, write_rds
from engine.recipe import load_recipe, new_recipe

EXPECTED_COLUMNS = ["X", "Y", "H", "DBH", "Species", "Latvus_h", "Latvus_d", "Status", "Type", "V"]
FLOAT_COLUMNS = ["X", "Y", "H", "DBH", "Latvus_h", "Latvus_d", "V"]
STRING_COLUMNS = ["Species", "Status", "Type"]


def small_forest(**kwargs):
    recipe = new_recipe()
    recipe["window"] = {"kind": "circle", "center": [0, 0], "radius": 60}
    recipe.update(kwargs)
    return generate(recipe)


class TableFormatTest(unittest.TestCase):
    def test_columns_and_dtypes(self):
        trees = small_forest().trees
        self.assertEqual(list(trees.columns), EXPECTED_COLUMNS)
        self.assertEqual(list(COLUMNS), EXPECTED_COLUMNS)
        for name in FLOAT_COLUMNS:
            self.assertEqual(trees[name].dtype, np.float64, name)
        for name in STRING_COLUMNS:
            self.assertTrue(pd.api.types.is_string_dtype(trees[name]), name)
            self.assertTrue(all(isinstance(v, str) for v in trees[name]), name)
        self.assertEqual(set(trees["Species"]), {"1", "2", "3"})
        self.assertEqual(set(trees["Status"]), {"Alive", "Dead"})
        self.assertEqual(set(trees["Type"]), {"ITD", "Simulated"})

    def test_rounding(self):
        trees = small_forest().trees
        for name, decimals in (("X", 2), ("Y", 2), ("H", 2), ("DBH", 2), ("Latvus_h", 2),
                               ("Latvus_d", 2), ("V", 4)):
            values = trees[name].dropna().to_numpy()
            np.testing.assert_allclose(values, np.round(values, decimals), atol=1e-9, err_msg=name)


class CsvTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_csv_text(self):
        trees = small_forest().trees
        path = os.path.join(self.tmp, "t.csv")
        write_csv(trees, path)
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        self.assertEqual(lines[0], "X,Y,H,DBH,Species,Latvus_h,Latvus_d,Status,Type,V")
        self.assertEqual(len(lines), len(trees) + 1)
        two = r"-?\d+\.\d{2}"
        pattern = re.compile(rf"^{two},{two},{two},{two},[123],({two}|NA),({two}|NA),"
                             rf"(Alive|Dead),(ITD|Simulated),\d+\.\d{{4}}$")
        for line in lines[1:]:
            self.assertRegex(line, pattern)
        simulated = [line for line in lines[1:] if ",Simulated," in line]
        self.assertTrue(simulated and all(",NA,NA," in line for line in simulated))

    def test_csv_round_trip(self):
        trees = small_forest().trees
        path = os.path.join(self.tmp, "t.csv")
        write_csv(trees, path)
        back = read_trees(path)
        pd.testing.assert_frame_equal(back.astype({c: object for c in STRING_COLUMNS}),
                                      trees.astype({c: object for c in STRING_COLUMNS}))

    def test_write_forest_files(self):
        forest = small_forest()
        base = os.path.join(self.tmp, "sub", "forest")
        written = write_forest(forest, base, ("csv", "rds"), report_text="report")
        names = sorted(os.path.basename(p) for p in written)
        self.assertEqual(names, ["forest-recipe.json", "forest-report.txt", "forest-truth.csv",
                                 "forest.csv", "forest.rds"])
        truth = pd.read_csv(base + "-truth.csv", dtype=str, keep_default_na=False)
        self.assertEqual(len(truth), len(forest.trees))
        self.assertEqual(list(truth.columns), ["layer_id", "layer_name", "layer_type", "preset"])
        recipe = load_recipe(base + "-recipe.json")
        self.assertEqual(recipe["seed"], forest.recipe["seed"])
        again = generate(recipe)
        pd.testing.assert_frame_equal(again.trees, forest.trees)


class RdsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.trees = small_forest().trees
        self.path = os.path.join(self.tmp, "forest.rds")
        write_rds(self.trees, self.path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pyreadr_round_trip(self):
        import pyreadr
        back = pyreadr.read_r(self.path)[None]
        self.assertEqual(list(back.columns), EXPECTED_COLUMNS)
        for name in STRING_COLUMNS:
            self.assertTrue(all(isinstance(v, str) for v in back[name]), name)
            self.assertEqual(list(back[name]), list(self.trees[name]))
        for name in FLOAT_COLUMNS:
            np.testing.assert_array_equal(np.isnan(back[name].to_numpy(float)),
                                          np.isnan(self.trees[name].to_numpy(float)))
            np.testing.assert_allclose(back[name].to_numpy(float), self.trees[name].to_numpy(float),
                                       equal_nan=True)
        via_reader = read_trees(self.path)
        self.assertTrue(all(isinstance(v, str) for v in via_reader["Species"]))

    def test_classes_in_r(self):
        rscript = shutil.which("Rscript")
        if rscript is None:
            self.skipTest("Rscript is not on the PATH; the R-side check was skipped.")
        code = (
            f'd <- readRDS("{self.path.replace(chr(92), "/")}");'
            'cat(is.data.frame(d), class(d), "\\n");'
            'cat(paste(names(d), sapply(d, class), sep=":"), "\\n");'
            'cat(sum(is.na(d$Latvus_h)), sum(is.nan(d$Latvus_h)), nrow(d), "\\n")'
        )
        out = subprocess.run([rscript, "-e", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.strip().splitlines()
        self.assertEqual(lines[0].split(), ["TRUE", "data.frame"])
        classes = dict(item.split(":") for item in lines[1].split())
        for name in FLOAT_COLUMNS:
            self.assertEqual(classes[name], "numeric", name)
        for name in STRING_COLUMNS:
            self.assertEqual(classes[name], "character", name)
        n_na, n_nan, n_rows = map(int, lines[2].split())
        self.assertEqual(n_rows, len(self.trees))
        self.assertEqual(n_na, int(self.trees["Latvus_h"].isna().sum()))
        self.assertGreater(n_na, 0)
        self.assertEqual(n_nan, 0, "missing values must be NA, not NaN")


class ReaderTest(unittest.TestCase):
    def test_reads_numeric_species_and_other_case(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "ext.csv")
            with open(path, "w") as fh:
                fh.write("x;y;dbh;species\n1.5;2.5;12.0;2\n3.0;4.0;20.0;1\n")
            trees = read_trees(path)
            self.assertEqual(list(trees.columns), EXPECTED_COLUMNS)
            self.assertEqual(list(trees["Species"]), ["2", "1"])
            self.assertTrue(trees["H"].isna().all())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_bad_file_is_a_clear_error(self):
        from engine.io import TreeFileError
        with self.assertRaises(TreeFileError):
            read_trees(os.path.join(ROOT, "no_such_file.csv"))
        with self.assertRaises(TreeFileError):
            read_trees(os.path.join(ROOT, "requirements.txt"))


if __name__ == "__main__":
    unittest.main()
