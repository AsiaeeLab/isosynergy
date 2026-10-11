"""Repository-relative inputs, outputs and execution limits."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
LIB = ROOT / "lib"
DATA = ROOT / "data"
SMOKE = os.environ.get("SIR_SMOKE", "0") == "1"
OUT = ROOT / "output"
if SMOKE:
    OUT /= "smoke"
INPUTS = OUT / "inputs"
REAL = OUT / "realdata"
SIMS = OUT / "simulations"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
LOGS = OUT / "logs"
WORKERS = max(1, min(16, int(os.environ.get("SIR_WORKERS", "16"))))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "ARROW_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
for directory in (INPUTS, REAL, SIMS, FIGURES, TABLES, LOGS, OUT / "cache"):
    directory.mkdir(parents=True, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = str(OUT / "cache" / "numba")
os.environ["MPLCONFIGDIR"] = str(OUT / "cache" / "matplotlib")
sys.dont_write_bytecode = True
for directory in (REPO / "inst" / "python", LIB, ROOT / "stages" / "02_realdata", ROOT / "stages" / "03_simulations", ROOT / "stages" / "04_figures_tables"):
    sys.path.insert(0, str(directory))


def raw_file(relative):
    """Prefer an existing public input; downloaded inputs stay in the output tree."""
    placed = REPO / "data" / "raw" / relative
    return placed if placed.is_file() else OUT / "raw" / relative
