"""Command line: generate, report, validate and example (no GUI, no images)."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import EXAMPLES_DIR  # also puts the project on sys.path
import synthforest


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = synthforest.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_example_list_and_print(self):
        code, out, _ = run("example")
        self.assertEqual(code, 0)
        self.assertIn("mosaic_with_trail", out)
        code, out, _ = run("example", "mosaic_with_trail")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["name"], "mosaic_with_trail")

    def test_generate_report_validate(self):
        recipe = os.path.join(EXAMPLES_DIR, "clearing_and_rows.json")
        base = os.path.join(self.tmp, "forest")
        code, out, err = run("generate", recipe, "--seed", "42", "--out", base, "--formats", "csv,rds")
        self.assertEqual(code, 0, err)
        for suffix in (".csv", ".rds", "-truth.csv", "-recipe.json", "-report.txt"):
            self.assertTrue(os.path.exists(base + suffix), suffix)
        code, out, err = run("report", base + ".csv")
        self.assertEqual(code, 0, err)
        self.assertIn("Clark-Evans", out)
        self.assertIn("from the recipe file", out)
        code, out, err = run("validate", recipe)
        self.assertEqual(code, 0, err)
        self.assertIn("OK", out)

    def test_generate_on_a_tree_file(self):
        base = os.path.join(self.tmp, "plot")
        code, _, err = run("generate", os.path.join(EXAMPLES_DIR, "default.json"), "--out", base)
        self.assertEqual(code, 0, err)
        recipe = os.path.join(self.tmp, "edit.json")
        with open(recipe, "w") as fh:
            json.dump({"name": "edited", "background": {"file": "plot.csv"},
                       "layers": [{"type": "clear", "shape": {"kind": "circle", "center": [0, 0],
                                                              "radius": 100}}]}, fh)
        code, out, err = run("generate", recipe, "--out", os.path.join(self.tmp, "edited"))
        self.assertEqual(code, 0, err)
        with open(os.path.join(self.tmp, "edited-recipe.json")) as fh:
            copy = json.load(fh)
        self.assertEqual(copy["background"]["file"], os.path.abspath(base + ".csv"))
        code, _, err = run("edit", os.path.join(self.tmp, "missing.csv"))
        self.assertEqual(code, 2)
        self.assertIn("not found", err)

    def test_errors_are_messages_not_tracebacks(self):
        bad = os.path.join(self.tmp, "bad.json")
        with open(bad, "w") as fh:
            json.dump({"layers": [{"type": "random", "params": {"density": -3},
                                   "shape": {"kind": "window"}}]}, fh)
        code, _, err = run("generate", bad)
        self.assertEqual(code, 2)
        self.assertIn('"params.density"', err)
        self.assertNotIn("Traceback", err)
        code, _, err = run("report", os.path.join(self.tmp, "missing.csv"))
        self.assertEqual(code, 2)
        self.assertIn("not found", err)
        code, _, err = run("generate", os.path.join(EXAMPLES_DIR, "default.json"), "--formats", "xls")
        self.assertEqual(code, 2)
        self.assertIn("unknown format", err)


if __name__ == "__main__":
    unittest.main()
