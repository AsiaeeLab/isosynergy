"""Synergy testing for drug-combination viability matrices.

Bliss independence is the reference, with control normalized to one. The primary
standardized excess test, companion mean test, region scan, replicate handling
and within-family BH screens share the same reference implementation."""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

from .core import fit  # noqa: E402
from .screening import screen, ScreenResult  # noqa: E402
from .noise import (ParametricNoise, NoiseBank, SDCurve, PlateVarianceTable, oneil_sd_curve,  # noqa: E402
                    sd_curve_from_replicates, seed_for)
from .multiplicity import bh_qvalues, bh_reject, empirical_null  # noqa: E402
from .result import FitResult, DataLog  # noqa: E402
from . import simulate  # noqa: E402

__version__ = "0.1.0"
__all__ = ["fit", "screen", "ScreenResult", "ParametricNoise", "NoiseBank", "SDCurve", "PlateVarianceTable",
           "oneil_sd_curve", "sd_curve_from_replicates", "seed_for", "bh_qvalues", "bh_reject", "empirical_null",
           "FitResult", "DataLog", "simulate"]
