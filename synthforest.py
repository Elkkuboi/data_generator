#!/usr/bin/env python3
"""synthforest: paint synthetic forests and export them as treemaps.

    python synthforest.py                                  # GUI
    python synthforest.py generate recipe.json --seed 42 --out forest --formats csv,rds,png
    python synthforest.py report forest.csv
    python synthforest.py view forest.rds
    python synthforest.py edit real_treemap.rds                # paint layers on a tree file
    python synthforest.py example mosaic_with_trail > my_recipe.json
    python synthforest.py validate recipe.json
"""

import argparse
import importlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(HERE, "examples")
MIN_PYTHON = (3, 10)
REQUIRED = (("numpy", "1.23"), ("scipy", "1.9"), ("shapely", "2.0"),
            ("matplotlib", "3.6"), ("pandas", "1.5"), ("pyreadr", "0.4.7"))
TABLE_FORMATS = ("csv", "rds")
IMAGE_FORMATS = ("png", "svg", "pdf")


def _version_tuple(text):
    return tuple(int(p) for p in re.findall(r"\d+", str(text))[:3])


def check_dependencies():
    """Exit with a clear message if Python or a required package is missing or too old."""
    if sys.version_info < MIN_PYTHON:
        sys.exit(f"synthforest needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer; "
                 f"this is Python {sys.version.split()[0]}.")
    missing, old = [], []
    for name, minimum in REQUIRED:
        try:
            module = importlib.import_module(name)
        except ImportError:
            missing.append(name)
            continue
        if _version_tuple(getattr(module, "__version__", "0")) < _version_tuple(minimum):
            old.append(f"{name}>={minimum}")
    if missing or old:
        lines = []
        if missing:
            lines.append("synthforest needs these Python packages, which are not installed: "
                         + ", ".join(missing))
            lines.append(f"    pip install {' '.join(missing)}")
        if old:
            lines.append("These packages are too old: " + ", ".join(old))
            lines.append(f"    pip install --upgrade {' '.join(repr(o) for o in old)}")
        lines.append("Or install everything at once:  pip install -r requirements.txt")
        sys.exit("\n".join(lines))


def example_names():
    """Names of the bundled example recipes."""
    if not os.path.isdir(EXAMPLES_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(EXAMPLES_DIR) if f.endswith(".json"))


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_generate(args):
    from engine.generate import generate
    from engine.io import write_forest
    from engine.recipe import load_recipe
    from engine.report import forest_report, format_report, window_description

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    unknown = [f for f in formats if f not in TABLE_FORMATS + IMAGE_FORMATS]
    if unknown:
        raise UsageError(f"unknown format(s): {', '.join(unknown)}; choose from "
                         + ", ".join(TABLE_FORMATS + IMAGE_FORMATS))
    recipe = load_recipe(args.recipe)
    forest = generate(recipe, seed=args.seed,
                      base_dir=os.path.dirname(os.path.abspath(args.recipe)))
    base = args.out or forest.recipe["name"]
    rep = forest_report(forest.trees, forest.window, window_description(forest.recipe["window"]))
    title = f"{os.path.basename(base)} (recipe {forest.recipe['name']}, seed {forest.recipe['seed']})"
    written = write_forest(forest, base, [f for f in formats if f in TABLE_FORMATS],
                           report_text=format_report(rep, title=title))
    images = [f for f in formats if f in IMAGE_FORMATS]
    if images:
        from gui.figures import save_map
        for fmt in images:
            path = f"{base}.{fmt}"
            save_map(path, forest.trees, forest.window, colour_by=args.colour_by,
                     truth=forest.truth, hide_coordinates=args.hide_coordinates,
                     title=None if args.hide_coordinates else title)
            written.append(path)
    n = len(forest.trees)
    print(f"{n:,} trees ({rep['density']:.0f} /ha) generated in {forest.elapsed:.2f} s "
          f"with seed {forest.recipe['seed']}.")
    for warning in forest.warnings:
        print(f"warning: {warning}")
    for path in written:
        print(f"  wrote {path}")
    return 0


def cmd_report(args):
    from engine.io import read_trees, window_for_trees
    from engine.report import forest_report, format_report, window_description

    trees = read_trees(args.file)
    window, spec, origin, _ = window_for_trees(args.file, trees)
    note = f"{window_description(spec)} ({origin})"
    print(format_report(forest_report(trees, window, note),
                        title=f"Report for {os.path.basename(args.file)}"))
    return 0


def cmd_validate(args):
    from engine.recipe import load_recipe

    recipe = load_recipe(args.recipe)
    print(f"{args.recipe}: OK ({len(recipe['layers'])} layers, seed {recipe['seed']}).")
    return 0


def cmd_example(args):
    names = example_names()
    if not args.name:
        print("Examples:\n  " + "\n  ".join(names))
        return 0
    if args.name not in names:
        raise UsageError(f"no example called {args.name!r}; available: {', '.join(names)}")
    with open(os.path.join(EXAMPLES_DIR, args.name + ".json"), encoding="utf-8") as fh:
        sys.stdout.write(fh.read())
    return 0


def cmd_view(args):
    return run_gui(view_file=args.file)


def cmd_edit(args):
    if not os.path.exists(args.file):
        raise UsageError(f"{args.file}: file not found.")
    return run_gui(edit_file=args.file)


def run_gui(view_file=None, recipe_file=None, edit_file=None):
    """Start the graphical interface."""
    try:
        importlib.import_module("tkinter")
    except ImportError:
        sys.exit("The GUI needs tkinter, which this Python does not include.\n"
                 "  Windows/macOS: install Python from python.org (tkinter is included).\n"
                 "  Debian/Ubuntu: sudo apt install python3-tk\n"
                 "The command line works without it: python synthforest.py --help")
    from gui.app import main as gui_main
    return gui_main(view_file=view_file, recipe_file=recipe_file, edit_file=edit_file)


class UsageError(Exception):
    """Bad command-line usage."""


def build_parser():
    parser = argparse.ArgumentParser(
        prog="synthforest", description="Paint synthetic forests; export treemaps.")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on errors")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("generate", help="generate a forest from a recipe")
    p.add_argument("recipe", help="recipe JSON file")
    p.add_argument("--seed", type=int, default=None, help="random seed (default: the recipe's)")
    p.add_argument("--out", default=None, help="output base name (default: the recipe's name)")
    p.add_argument("--formats", default="csv",
                   help="comma-separated: csv, rds, png, svg, pdf (default csv)")
    p.add_argument("--colour-by", default="species", choices=("species", "status", "type", "layer"),
                   help="colour of trees in images")
    p.add_argument("--hide-coordinates", action="store_true", help="no axis ticks in images")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("report", help="realism report of a CSV or RDS tree file")
    p.add_argument("file")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("view", help="open a CSV or RDS tree file in the viewer")
    p.add_argument("file")
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("edit", help="open a CSV or RDS tree file in the editor to paint on it")
    p.add_argument("file")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("example", help="print an example recipe (no name: list them)")
    p.add_argument("name", nargs="?")
    p.set_defaults(func=cmd_example)

    p = sub.add_parser("validate", help="check a recipe and list every problem")
    p.add_argument("recipe")
    p.set_defaults(func=cmd_validate)
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    check_dependencies()
    sys.path.insert(0, HERE)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        return run_gui()
    from engine.io import TreeFileError
    from engine.recipe import RecipeError
    try:
        return args.func(args)
    except RecipeError as exc:
        print("The recipe has problems:", file=sys.stderr)
        for message in exc.errors:
            print(f"  - {message}", file=sys.stderr)
        return 2
    except (TreeFileError, UsageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc.strerror or exc}: {exc.filename or ''}", file=sys.stderr)
        return 1
    except Exception as exc:        # last resort: never dump a traceback on users
        if args.debug:
            raise
        print(f"unexpected error: {type(exc).__name__}: {exc}\n"
              "Run again with --debug for details.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
