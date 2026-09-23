"""Tree attributes: species, dbh, height, volume, crown, status and type.

All functions are vectorised over trees.  Species are indexed 0, 1, 2 =
pine, spruce, broadleaved.

Height model (Näslund):  h = 1.3 + d^2 / (a + b*d)^2  with d in cm, h in m.
As d grows, h approaches 1.3 + 1/b^2, so b is fixed by a height ceiling per
species and a is solved so that the median dbh of the default size mix maps
to the median height of the defaults table:

    b = 1 / sqrt(ceiling - 1.3)
    a = d_med / sqrt(h_med - 1.3) - b * d_med

    species      ceiling  d_med  h_med    a        b
    pine          30 m     15    14 m   1.4092   0.18666
    spruce        33 m     12    11 m   1.7216   0.17761
    broadleaved   27 m     12    13 m   1.1411   0.19726

Because h is monotone in d and the noise has median 1, the median height of
the default forest reproduces the table.
"""

from functools import lru_cache
from typing import NamedTuple

import numpy as np
from scipy import optimize, special

from .presets import (
    DBH_MAX,
    DBH_MIN,
    DEFAULT_MEDIAN_DBH,
    DEFAULT_MEDIAN_HEIGHT,
)

HEIGHT_CEILING = (30.0, 33.0, 27.0)
NASLUND_B = np.array([1.0 / np.sqrt(c - 1.3) for c in HEIGHT_CEILING])
NASLUND_A = np.array([
    d / np.sqrt(h - 1.3) - b * d
    for d, h, b in zip(DEFAULT_MEDIAN_DBH, DEFAULT_MEDIAN_HEIGHT, NASLUND_B)
])
HEIGHT_NOISE_SD = 0.08            # log-scale, multiplies (h - 1.3)

FORM_FACTOR = np.array([0.51, 0.52, 0.48])
CROWN_DIAMETER_PER_DBH = np.array([0.11, 0.12, 0.12])   # m per cm
CROWN_DIAMETER_NOISE_SD = 0.20    # log-scale
CROWN_RATIO = np.array([0.54, 0.69, 0.58])              # crown length / height
CROWN_RATIO_NOISE_SD = 0.12       # log-scale
CROWN_RATIO_MIN = 0.05

# Laser detection: logistic in dbh, 50 % at 17 cm and 90 % at 25 cm.
ITD_DBH50 = 17.0
ITD_DBH90 = 25.0
ITD_SCALE = (ITD_DBH90 - ITD_DBH50) / np.log(9.0)


class Composition(NamedTuple):
    """Resolved composition of an adding layer (one entry per species)."""

    shares: tuple      # species shares by stems, sum 1
    dbh: tuple         # (median cm, log-sigma) per species
    dead: tuple        # probability of "Dead" per species


def _truncated_std_normal(a, b, u):
    """Inverse-CDF draw from N(0, 1) truncated to [a, b], accurate in both tails."""
    if a > 0:
        return -_truncated_std_normal(-b, -a, 1.0 - u)
    pa, pb = special.ndtr(a), special.ndtr(b)
    return np.clip(special.ndtri(pa + u * (pb - pa)), a, b)


def _log_bounds(mu, sigma):
    return (np.log(DBH_MIN) - mu) / sigma, (np.log(DBH_MAX) - mu) / sigma


@lru_cache(maxsize=512)
def lognormal_mu(median, sigma):
    """Return mu so that LogNormal(mu, sigma) truncated to the dbh range has this median.

    Raises ValueError if the median cannot be reached (too close to a limit).
    """
    target = np.log(median)

    def excess(mu):
        return mu + sigma * _truncated_std_normal(*_log_bounds(mu, sigma), 0.5) - target

    lo, hi = np.log(DBH_MIN) - 12 * sigma, np.log(DBH_MAX) + 12 * sigma
    try:
        return float(optimize.brentq(excess, lo, hi, xtol=1e-10))
    except ValueError:
        raise ValueError(
            f"median dbh {median:g} cm is not reachable with sigma {sigma:g} "
            f"inside {DBH_MIN:g}-{DBH_MAX:g} cm") from None


def sample_dbh(median, sigma, u):
    """Transform uniforms ``u`` into truncated log-normal dbh values (cm)."""
    mu = lognormal_mu(float(median), float(sigma))
    a, b = _log_bounds(mu, sigma)
    return np.exp(mu + sigma * _truncated_std_normal(a, b, u))


def naslund_height(dbh, species):
    """Noise-free Näslund height (m) for dbh in cm."""
    a, b = NASLUND_A[species], NASLUND_B[species]
    return 1.3 + dbh ** 2 / (a + b * dbh) ** 2


def stem_volume(dbh, height, species):
    """Stem volume (m^3) = form factor * basal area * height."""
    return FORM_FACTOR[species] * np.pi * (dbh / 200.0) ** 2 * height


def itd_probability(dbh):
    """Probability that laser scanning detects a tree individually."""
    return 1.0 / (1.0 + np.exp(-(dbh - ITD_DBH50) / ITD_SCALE))


def assign_attributes(n, composition, rng, laser_artefacts=True):
    """Draw attributes for ``n`` trees of one layer.

    Returns a dict of arrays: species (0..2), dbh, height, crown_d, crown_l,
    dead (bool), itd (bool).  Volume is computed later from rounded values.
    The order of random draws is fixed, so results are reproducible.
    """
    shares = np.asarray(composition.shares, dtype=float)
    species = rng.choice(3, size=n, p=shares / shares.sum()).astype(np.int8)

    u = rng.random(n)
    dbh = np.empty(n)
    for s in range(3):
        mask = species == s
        if mask.any():
            median, sigma = composition.dbh[s]
            dbh[mask] = sample_dbh(median, sigma, u[mask])

    dbh = np.round(dbh, 2)          # the measured value, as written to the table
    height = 1.3 + (naslund_height(dbh, species) - 1.3) * np.exp(
        rng.normal(0.0, HEIGHT_NOISE_SD, n))
    crown_d = CROWN_DIAMETER_PER_DBH[species] * dbh * np.exp(
        rng.normal(0.0, CROWN_DIAMETER_NOISE_SD, n))
    ratio = CROWN_RATIO[species] * np.exp(rng.normal(0.0, CROWN_RATIO_NOISE_SD, n))
    crown_l = np.clip(ratio, CROWN_RATIO_MIN, 1.0) * height
    dead = rng.random(n) < np.asarray(composition.dead)[species]
    detect = rng.random(n)
    itd = detect < itd_probability(dbh) if laser_artefacts else np.ones(n, dtype=bool)
    return {"species": species, "dbh": dbh, "height": height, "crown_d": crown_d,
            "crown_l": crown_l, "dead": dead, "itd": itd}
