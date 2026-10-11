"""Result containers."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DataLog:
    """What the data-handling rules did to one matrix."""
    design: str = ""
    n_replicates: int = 1
    input_shape: tuple = ()
    analyzed_shape: tuple = ()
    rows_kept: List[int] = field(default_factory=list)   # original dose indices of drug A (row 0 = zero dose)
    cols_kept: List[int] = field(default_factory=list)
    dropped_doses: List[Dict[str, Any]] = field(default_factory=list)       # rule R
    missing_cells: List[tuple] = field(default_factory=list)                # original (i, j) of missing combination wells
    partial_replicates: int = 0                                             # wells with some but not all replicates
    screened: List[Dict[str, Any]] = field(default_factory=list)            # replicate screen drops
    screen_applied: bool = False
    reference_flags_per_plate: float = 0.0
    notes: List[str] = field(default_factory=list)

    def note(self, msg):
        self.notes.append(msg)


@dataclass
class FitResult:
    design: str
    alpha: float
    # descriptive
    activity: float
    activity_all_wells: float
    a_hat: np.ndarray
    b_hat: np.ndarray
    fitted_surface: np.ndarray        # 2D isotonic fit P_M(Y) on the analyzed grid (NaN at missing wells)
    excess_map: np.ndarray            # isotonic excess a_hat_i b_hat_j - P_M(Y)_ij (descriptive)
    raw_excess_map: np.ndarray        # Y_i0 Y_0j - Y_ij
    analyzed_matrix: np.ndarray
    doses_a: Optional[np.ndarray]
    doses_b: Optional[np.ndarray]
    # companion (mean) test
    mean_excess: float
    mean_excess_se: float
    p_mean: float
    mean_test: Dict[str, Any]
    mean_tests: Dict[str, Dict[str, Any]]
    p_mean_boot: float
    # primary test
    t_std: float
    p_tstd: float
    p_tstd_stage1: float
    # localization
    p_scan: float
    p_scan_stage1: float
    regions: List[Dict[str, Any]]
    stage: int
    B_used: int
    # fragility and data handling
    fragility: Dict[str, Any]
    log: DataLog
    settings: Dict[str, Any]

    @property
    def claims(self):
        """Regions claimed at level alpha (familywise error control over region claims)."""
        return [r for r in self.regions if r["claimed"]]

    @property
    def call(self):
        return bool(self.p_tstd <= self.alpha)

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, DataLog):
                d[k] = dict(v.__dict__)
            else:
                d[k] = v
        return d

    def summary(self):
        L = []
        L.append(f"sirsyn fit: design {self.design}, analyzed grid {self.log.analyzed_shape} "
                 f"(input {self.log.input_shape}, {self.log.n_replicates} replicate(s))")
        L.append(f"  activity (mean 1 - Y over combination wells) {self.activity:.4f}")
        L.append(f"  mean Bliss excess {self.mean_excess:.4f} (SE {self.mean_excess_se:.4f}); "
                 f"{self.mean_test.get('name', '')}: p = {self.p_mean:.4g}")
        L.append(f"  T_std = {self.t_std:.4g}, p = {self.p_tstd:.4g} (stage {self.stage}, B = {self.B_used}; "
                 f"stage 1 p = {self.p_tstd_stage1:.4g})")
        L.append(f"  region scan: global p = {self.p_scan:.4g}; {len(self.claims)} region(s) claimed at {self.alpha}")
        for r in self.claims:
            L.append(f"    region {r['label']} ({r['type']}), adjusted p = {r['p_adj']:.4g}")
        if self.fragility.get("computed"):
            f = self.fragility
            L.append(f"  fragility: most influential cell {f['loco_cell']} (p without it {f['loco_p']:.4g}, "
                     f"flag {f['loco_flag']}); most influential single-agent dose {f['sa_dose']} "
                     f"(p without it {f['sa_p']:.4g}, flag {f['sa_flag']})")
        lg = self.log
        if lg.dropped_doses:
            L.append(f"  rule R dropped: {lg.dropped_doses}")
        if lg.missing_cells:
            L.append(f"  missing combination wells (original indices): {lg.missing_cells}")
        if lg.screen_applied:
            L.append(f"  replicate screen dropped {len(lg.screened)} replicate value(s)")
        for n in lg.notes:
            L.append(f"  note: {n}")
        return "\n".join(L)
