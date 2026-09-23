"""Stand-type presets and the defaults taken from the real treemaps.

Everything a recipe leaves unset falls back to the values here.  Species are
indexed 0, 1, 2 = pine, spruce, broadleaved throughout the engine; the codes
written to the tree table are "1", "2", "3".
"""

SPECIES = ("pine", "spruce", "broadleaved")
SPECIES_CODES = ("1", "2", "3")

# --- Defaults table (approximate values from the real data) ----------------

DEFAULT_WINDOW = {"kind": "circle", "center": [0.0, 0.0], "radius": 306.0}
DEFAULT_DENSITY = 650.0                         # trees / ha
DEFAULT_SHARES = (0.27, 0.43, 0.30)             # by stems
DEFAULT_MEDIAN_DBH = (15.0, 12.0, 12.0)         # cm
DEFAULT_DBH_SIGMA = 0.70                        # log-scale spread of the default size mix
DEFAULT_MEDIAN_HEIGHT = (14.0, 11.0, 13.0)      # m
DEFAULT_DEAD_SHARE = (0.060, 0.021, 0.036)
DEFAULT_IMPUTED_SHARE = 0.60
DBH_MIN, DBH_MAX = 4.5, 150.0                   # cm

DEFAULT_BACKGROUND = {
    "enabled": True,
    "type": "random",
    "params": {"density": DEFAULT_DENSITY},
    "composition": {},
}

# --- Size classes: (median dbh in cm, log-scale sigma), all species --------

SIZE_CLASSES = {
    "seedling": (5.5, 0.15),
    "young": (9.0, 0.30),
    "middle": (16.0, 0.30),
    "mature": (26.0, 0.28),
    "old": (36.0, 0.45),
}

# --- Default parameters of each layer type ---------------------------------

TYPE_DEFAULTS = {
    "random": {"density": DEFAULT_DENSITY},
    "clustered": {"cluster_density": 40.0, "trees_per_cluster": 16.0, "cluster_radius": 5.0},
    "regular": {"density": DEFAULT_DENSITY, "min_distance": 2.5},
    "rows": {"row_spacing": 3.0, "tree_spacing": 2.0, "angle": 0.0, "jitter": 0.3},
    "gradient": {"density_start": 200.0, "density_end": 1200.0, "direction": 0.0},
    "clear": {},
    "thin": {"fraction": 0.3},
}

# --- Stand-type presets ----------------------------------------------------
# A preset supplies a pattern type, its parameters and a composition.  Keys
# set explicitly in a layer override the preset.

PRESETS = {
    "young_spruce": {
        "description": "dense, small, mostly spruce, weakly clustered",
        "type": "clustered",
        "params": {"cluster_density": 90.0, "trees_per_cluster": 20.0, "cluster_radius": 8.0},
        "composition": {
            "species": {"pine": 0.05, "spruce": 0.85, "broadleaved": 0.10},
            "size_class": "young",
            "dead_share": 0.01,
        },
    },
    "pine_heath": {
        "description": "sparse to medium, pine-dominated, fairly regular",
        "type": "regular",
        "params": {"density": 450.0, "min_distance": 2.8},
        "composition": {
            "species": {"pine": 0.85, "spruce": 0.10, "broadleaved": 0.05},
            "size_class": "middle",
            "dead_share": 0.03,
        },
    },
    "mixed_birch_spruce": {
        "description": "medium density, mixed, birch clustered",
        "type": "clustered",
        "params": {"cluster_density": 30.0, "trees_per_cluster": 25.0, "cluster_radius": 6.0},
        "composition": {
            "species": {"pine": 0.10, "spruce": 0.50, "broadleaved": 0.40},
            "size_class": "middle",
        },
    },
    "old_growth": {
        "description": "wide size range, large trees, many standing dead",
        "type": "clustered",
        "params": {"cluster_density": 30.0, "trees_per_cluster": 14.0, "cluster_radius": 8.0},
        "composition": {
            "species": {"pine": 0.25, "spruce": 0.55, "broadleaved": 0.20},
            "dbh": {"median": 22.0, "sigma": 0.8},
            "dead_share": 0.18,
        },
    },
    "seedling_stand": {
        "description": "very dense, very small",
        "type": "random",
        "params": {"density": 3000.0},
        "composition": {
            "species": {"pine": 0.30, "spruce": 0.30, "broadleaved": 0.40},
            "size_class": "seedling",
            "dead_share": 0.01,
        },
    },
    "managed_rows": {
        "description": "planted rows, uniform size",
        "type": "rows",
        "params": {"row_spacing": 3.0, "tree_spacing": 2.2, "angle": 0.0, "jitter": 0.3},
        "composition": {
            "species": {"pine": 0.90, "spruce": 0.10, "broadleaved": 0.0},
            "dbh": {"median": 14.0, "sigma": 0.12},
            "dead_share": 0.01,
        },
    },
}


def preset_names():
    """Names of all presets, in display order."""
    return list(PRESETS)
