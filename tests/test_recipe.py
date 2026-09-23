"""Recipe validation, presets, saving and loading, generators, examples."""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import EXAMPLES_DIR
from engine.generate import generate
from engine.generators import GENERATORS, apply_generator
from engine.presets import PRESETS
from engine.recipe import (
    RecipeError,
    load_recipe,
    new_recipe,
    parse_recipe,
    recipe_to_json,
    resolve_layer,
    save_recipe,
    validate_recipe,
)


def one_layer(**layer):
    recipe = new_recipe()
    layer.setdefault("shape", {"kind": "window"})
    recipe["layers"] = [layer]
    return recipe


class ValidationTest(unittest.TestCase):
    def errors_of(self, recipe):
        with self.assertRaises(RecipeError) as ctx:
            validate_recipe(recipe)
        return ctx.exception.errors

    def test_empty_recipe_is_valid(self):
        recipe = validate_recipe({})
        self.assertEqual(recipe["window"]["kind"], "circle")
        self.assertEqual(recipe["window"]["radius"], 306.0)
        self.assertTrue(recipe["background"]["enabled"])
        self.assertEqual(recipe["layers"], [])

    def test_unknown_key_names_layer_and_key(self):
        errors = self.errors_of(one_layer(id="L7", name="Patch", type="random",
                                          params={"densty": 100}))
        self.assertEqual(len(errors), 1)
        self.assertIn('layer 1 "Patch" (L7)', errors[0])
        self.assertIn('"params.densty"', errors[0])
        self.assertIn('did you mean "params.density"', errors[0])

    def test_unknown_top_level_key(self):
        errors = self.errors_of({"layerz": []})
        self.assertIn('"layerz"', errors[0])

    def test_negative_density(self):
        errors = self.errors_of(one_layer(type="random", params={"density": -5}))
        self.assertIn('"params.density"', errors[0])
        self.assertIn("Suggestion", errors[0])

    def test_min_distance_incompatible_with_density(self):
        errors = self.errors_of(one_layer(type="regular",
                                          params={"density": 5000, "min_distance": 2.5}))
        self.assertIn('"params.density"', errors[0])
        self.assertIn("min_distance", errors[0])
        self.assertIn("Suggestion", errors[0])

    def test_shares_must_sum_to_one(self):
        errors = self.errors_of(one_layer(type="random", composition={
            "species": {"pine": 0.5, "spruce": 0.3, "broadleaved": 0.1}}))
        self.assertIn('"composition.species"', errors[0])
        self.assertIn("sum to 0.900", errors[0])
        self.assertIn("pine 0.556", errors[0])

    def test_several_errors_reported_together(self):
        recipe = new_recipe()
        recipe["layers"] = [
            {"type": "random", "params": {"density": -1}, "shape": {"kind": "window"}},
            {"type": "clear", "shape": {"kind": "circle", "center": [0, 0]}},
            {"type": "thin", "params": {"fraction": 2}, "shape": {"kind": "window"}},
        ]
        errors = self.errors_of(recipe)
        self.assertEqual(len(errors), 3)
        self.assertTrue(errors[0].startswith("layer 1"))
        self.assertTrue(errors[1].startswith("layer 2") and "shape.radius" in errors[1])
        self.assertTrue(errors[2].startswith("layer 3") and "params.fraction" in errors[2])

    def test_type_specific_keys(self):
        errors = self.errors_of(one_layer(type="clear", composition={}))
        self.assertIn('not used by type "clear"', errors[0])
        errors = self.errors_of(one_layer(type="random", mode="paint"))
        self.assertIn('"mode"', errors[0])

    def test_bad_values(self):
        for layer, key in (
            (dict(type="random", enabled="yes"), "enabled"),
            (dict(type="random", preset="old_grwth"), "preset"),
            (dict(type="random", composition={"size_class": "huge"}), "composition.size_class"),
            (dict(type="random", composition={"size_class": "young", "dbh": {"median": 10, "sigma": 0.3}}),
             "composition.dbh"),
            (dict(type="thin", params={"fraction": 0.3, "dbh_below": 10, "dbh_above": 20}),
             "params.dbh_above"),
            (dict(type="random", shape={"kind": "polygon", "vertices": [[0, 0], [1, 1], [2, 2]]}),
             "shape.vertices"),
        ):
            with self.subTest(key=key):
                errors = self.errors_of(one_layer(**layer))
                self.assertIn(f'"{key}"', errors[0])

    def test_duplicate_ids(self):
        recipe = new_recipe()
        recipe["layers"] = [{"id": "A", "type": "clear", "shape": {"kind": "window"}},
                            {"id": "A", "type": "clear", "shape": {"kind": "window"}}]
        self.assertIn('"id"', self.errors_of(recipe)[0])

    def test_too_many_trees(self):
        errors = self.errors_of(one_layer(type="random", params={"density": 20000},
                                          shape={"kind": "window"}) | {
            "window": {"kind": "circle", "center": [0, 0], "radius": 5000}})
        self.assertIn("limit", errors[0])

    def test_invalid_json_message(self):
        with self.assertRaises(RecipeError) as ctx:
            parse_recipe('{"name": "x",\n "layers": [}', source="bad.json")
        self.assertIn("bad.json", str(ctx.exception))
        self.assertIn("line 2", str(ctx.exception))


class PresetTest(unittest.TestCase):
    def test_every_preset_generates(self):
        for name, preset in PRESETS.items():
            recipe = one_layer(preset=name, mode="replace",
                               shape={"kind": "circle", "center": [0, 0], "radius": 50})
            recipe = validate_recipe(recipe)
            with self.subTest(preset=name):
                self.assertEqual(recipe["layers"][0]["type"], preset["type"])
                trees = generate(recipe).trees
                self.assertGreater(len(trees), 50)

    def test_explicit_keys_override_preset(self):
        layer = validate_recipe(one_layer(preset="young_spruce", params={"cluster_radius": 2},
                                          composition={"dead_share": 0.5}))["layers"][0]
        resolved = resolve_layer(layer)
        self.assertEqual(resolved.params["cluster_radius"], 2)
        self.assertEqual(resolved.params["cluster_density"], 90)
        self.assertEqual(resolved.composition.dead, (0.5, 0.5, 0.5))
        self.assertAlmostEqual(resolved.composition.shares[1], 0.85)

    def test_size_override_replaces_preset_size(self):
        layer = validate_recipe(one_layer(preset="old_growth",
                                          composition={"size_class": "young"}))["layers"][0]
        self.assertEqual(resolve_layer(layer).composition.dbh[0], (9.0, 0.30))


class FileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_and_load_round_trip(self):
        recipe = validate_recipe(one_layer(type="clear", shape={
            "kind": "corridor", "points": [[0, 0], [10.5, 3.25], [40, -2]], "width": 3}))
        path = os.path.join(self.tmp, "r.json")
        save_recipe(recipe, path)
        self.assertEqual(load_recipe(path), recipe)
        with open(path, encoding="utf-8") as fh:
            self.assertIn("[10.5, 3.25]", fh.read())

    def test_missing_file(self):
        with self.assertRaises(RecipeError):
            load_recipe(os.path.join(self.tmp, "none.json"))


class GeneratorTest(unittest.TestCase):
    def test_each_generator_adds_editable_layers(self):
        for name in GENERATORS:
            rng = np.random.default_rng(3)
            recipe = apply_generator(new_recipe(), name, rng)
            with self.subTest(generator=name):
                self.assertGreater(len(recipe["layers"]), 0)
                self.assertEqual(len({layer["id"] for layer in recipe["layers"]}),
                                 len(recipe["layers"]))
                parse_recipe(recipe_to_json(recipe))
                self.assertGreater(len(generate(recipe).trees), 1000)

    def test_generators_are_deterministic(self):
        a = apply_generator(new_recipe(), "mosaic", np.random.default_rng(1))
        b = apply_generator(new_recipe(), "mosaic", np.random.default_rng(1))
        self.assertEqual(a, b)

    def test_mosaic_covers_window(self):
        recipe = apply_generator(new_recipe(), "mosaic", np.random.default_rng(4), n_patches=6)
        forest = generate(recipe)
        self.assertEqual((forest.truth["layer_id"] == "background").sum(), 0)
        self.assertTrue(all(layer["mode"] == "replace" for layer in recipe["layers"]))


class ExamplesTest(unittest.TestCase):
    def test_every_example_loads_and_generates(self):
        names = sorted(f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".json"))
        for required in ("default", "mosaic_with_trail", "clearing_and_rows", "old_growth_patch"):
            self.assertIn(required + ".json", names)
        for name in names:
            with self.subTest(example=name):
                recipe = load_recipe(os.path.join(EXAMPLES_DIR, name))
                forest = generate(recipe)
                self.assertGreater(len(forest.trees), 1000)
                self.assertTrue(os.path.exists(os.path.join(EXAMPLES_DIR, name[:-5] + ".png")))
                with open(os.path.join(EXAMPLES_DIR, name), encoding="utf-8") as fh:
                    json.load(fh)


if __name__ == "__main__":
    unittest.main()
