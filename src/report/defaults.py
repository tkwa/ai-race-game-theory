"""Default parameter values for the full AI race game theory model."""

import numpy as np

LAB_NAMES = ["OAI", "Ant", "GDM", "xAI", "China"]

# Resource shares (sum to 1)
R = np.array([0.25, 0.15, 0.30, 0.15, 0.15])

# Amity matrix: A[i,j] = how much lab i values lab j's aligned ASI
A = np.array(
    [
        [1.0, 0.4, 0.4, 0.3, 0.0],  # OAI
        [0.5, 1.0, 0.5, 0.3, 0.1],  # Ant
        [0.4, 0.4, 1.0, 0.3, 0.0],  # GDM
        [0.2, 0.2, 0.2, 1.0, 0.0],  # xAI
        [0.3, 0.3, 0.3, 0.2, 1.0],  # China
    ]
)

# Calibrated so that 1% spending -> 80% aligned, 50% spending -> 98% aligned
K = 33.9
ALPHA = 0.466

W = 2.0  # Winner-take-all exponent
DELTA = 0.5  # Public good parameter
RHO = 0.5  # Alignment correlation
Z = 1.0  # Misaligned AI power advantage

# Sensitivity analysis: triangular distributions (min, mode, max)
PARAM_RANGES = {
    "w": (1.0, 2.0, 5.0),
    "delta": (0.0, 0.5, 0.75),
    "rho": (0.0, 0.5, 1.0),
    "z": (0.5, 1.0, 1.0),
    "china_r": (0.05, 0.15, 0.25),
    "amity_scale": (0.67, 1.0, 2.0),  # <1 dilates away from 1, >1 shrinks toward 1
}
