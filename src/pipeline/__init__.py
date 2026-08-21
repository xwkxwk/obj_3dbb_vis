"""Shared constants for the unified pipeline"""

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOXER_ROOT = PROJECT_ROOT / "libs" / "boxer"
FIXED_WRITE_NAME = "boxer"
VIS_PANEL_ASPECT = 16.0 / 9.0
R_ALIGN = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0],
], dtype=np.float64)
R_ALIGN_INV = R_ALIGN.T

if str(BOXER_ROOT) not in sys.path:
    sys.path.insert(0, str(BOXER_ROOT))
