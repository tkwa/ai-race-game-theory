"""Default parameter values for the full AI race game theory model."""

import numpy as np

ANONYMIZE = False

_REAL_NAMES = ["OAI", "Ant", "GDM", "xAI", "China"]
_ANON_NAMES = ["Lab A", "Lab B", "Lab C", "Lab D", "Lab E"]


def get_lab_names() -> list[str]:
    """Return lab names, anonymized if ANONYMIZE is set."""
    return list(_ANON_NAMES if ANONYMIZE else _REAL_NAMES)


def anonymize_label(text: str) -> str:
    """Replace real lab names with anonymous labels in a string."""
    if not ANONYMIZE:
        return text
    # Order matters: replace longer strings first to avoid partial matches
    replacements = [
        ("Anthropic", "Lab B"),
        ("China", "Lab E"),
        ("USG", "Bloc 1 gov't"),
        ("US→", "Bloc 1→"),
        ("→US", "→Bloc 1"),
        ("OAI", "Lab A"),
        ("Ant", "Lab B"),
        ("GDM", "Lab C"),
        ("xAI", "Lab D"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


# For backward compatibility — but prefer get_lab_names()
LAB_NAMES = _REAL_NAMES

# Resource shares (sum to 1) — Tao's estimates
R = np.array([0.27, 0.27, 0.27, 0.09, 0.10])

# Amity matrix: A[i,j] = how much lab i values lab j's aligned ASI — Tao's estimates
#                         OAI   Ant   GDM   xAI   China
A = np.array(
    [
        [1.00, 0.20, 0.20, 0.15, 0.00],  # OAI
        [0.45, 1.00, 0.60, 0.30, 0.10],  # Ant
        [0.50, 0.65, 1.00, 0.40, 0.10],  # GDM (owns some of Anthropic)
        [0.20, 0.20, 0.20, 1.00, -0.20],  # xAI
        [0.30, 0.30, 0.30, 0.20, 1.00],  # China
    ]
)

# Calibrated so that 1% spending -> 80% aligned, 50% spending -> 98% aligned
K = 33.9
ALPHA = 0.466

W = 2.0  # Winner-take-all exponent
DELTA = 0.2  # Public good parameter
RHO = 0.5  # Alignment correlation
Z = 0.9  # Misaligned AI power advantage

# Sensitivity analysis: triangular distributions (min, mode, max)
PARAM_RANGES = {
    "w": (1.0, 2.0, 5.0),
    "delta": (0.0, 0.2, 0.75),
    "rho": (0.0, 0.5, 1.0),
    "z": (0.5, 0.9, 1.0),
    "china_r": (0.05, 0.10, 0.25),
    "amity_scale": (0.67, 1.0, 2.0),  # <1 dilates away from 1, >1 shrinks toward 1
}
