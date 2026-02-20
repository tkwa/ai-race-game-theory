"""Monte Carlo sensitivity analysis over game-theoretic parameters."""

import multiprocessing

import numpy as np

from report import defaults, primitives


def _compute_one_sample(params: dict) -> float:
    """Compute Nash equilibrium and expected human share for one parameter vector."""
    R = np.array(params["R"])
    A = np.array(params["A"])
    s_nash = primitives.full_model_nash(
        R=R,
        A=A,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=params["w"],
        z=params["z"],
        delta=params["delta"],
        rho=params["rho"],
    )
    return primitives.expected_human_share(
        s_all=s_nash,
        R=R,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=params["w"],
        z=params["z"],
        delta=params["delta"],
        rho=params["rho"],
    )


def _make_R(china_r: float) -> np.ndarray:
    """Build resource vector with given China share, rescaling US labs proportionally."""
    us_total = 1.0 - china_r
    us_base = defaults.R[:4]
    R = np.empty(5)
    R[:4] = us_base * (us_total / us_base.sum())
    R[4] = china_r
    return R


def _make_A(amity_scale: float) -> np.ndarray:
    """Scale off-diagonal amity: >1 shrinks toward 1, <1 dilates away from 1."""
    off_diag = defaults.A - np.eye(len(defaults.A))
    scaled = off_diag * amity_scale
    return scaled + np.eye(len(defaults.A))


def run_sensitivity(n_samples: int = 500, seed: int = 42, n_workers: int | None = None) -> dict:
    """Run Monte Carlo sensitivity analysis over w, δ, ρ, z, China R, amity scale.

    Samples parameters from triangular distributions, computes Nash
    equilibrium and expected human share for each, then buckets by
    parameter decile to compute conditional medians.
    """
    np.random.seed(seed)

    # Extract parameter ranges: (min, mode, max)
    w_range = defaults.PARAM_RANGES["w"]
    delta_range = defaults.PARAM_RANGES["delta"]
    rho_range = defaults.PARAM_RANGES["rho"]
    z_range = defaults.PARAM_RANGES["z"]
    china_r_range = defaults.PARAM_RANGES["china_r"]
    amity_range = defaults.PARAM_RANGES["amity_scale"]

    # Sample from triangular distributions
    w_samples = np.random.triangular(w_range[0], w_range[1], w_range[2], n_samples)
    delta_samples = np.random.triangular(delta_range[0], delta_range[1], delta_range[2], n_samples)
    rho_samples = np.random.triangular(rho_range[0], rho_range[1], rho_range[2], n_samples)
    z_samples = np.random.triangular(z_range[0], z_range[1], z_range[2], n_samples)
    china_r_samples = np.random.triangular(
        china_r_range[0], china_r_range[1], china_r_range[2], n_samples
    )
    amity_samples = np.random.triangular(amity_range[0], amity_range[1], amity_range[2], n_samples)

    params_list = [
        {
            "w": float(w_samples[i]),
            "delta": float(delta_samples[i]),
            "rho": float(rho_samples[i]),
            "z": float(z_samples[i]),
            "R": _make_R(float(china_r_samples[i])).tolist(),
            "A": _make_A(float(amity_samples[i])).tolist(),
            "china_r": float(china_r_samples[i]),
            "amity_scale": float(amity_samples[i]),
        }
        for i in range(n_samples)
    ]

    # Parallel computation across CPU cores
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 8)
    with multiprocessing.Pool(n_workers) as pool:
        human_shares = pool.map(_compute_one_sample, params_list)

    # Bucket each parameter into deciles and compute conditional medians
    w_buckets, w_medians = compute_median_by_buckets(list(w_samples), human_shares, n_buckets=10)
    delta_buckets, delta_medians = compute_median_by_buckets(
        list(delta_samples), human_shares, n_buckets=10
    )
    rho_buckets, rho_medians = compute_median_by_buckets(
        list(rho_samples), human_shares, n_buckets=10
    )
    z_buckets, z_medians = compute_median_by_buckets(list(z_samples), human_shares, n_buckets=10)
    china_buckets, china_medians = compute_median_by_buckets(
        list(china_r_samples), human_shares, n_buckets=10
    )
    # Compute average off-diagonal amity for labeling
    amity_avgs = [float(np.mean(_make_A(s)[~np.eye(5, dtype=bool)])) for s in amity_samples]
    amity_buckets, amity_medians = compute_median_by_buckets(amity_avgs, human_shares, n_buckets=10)

    variables = [
        (w_buckets, w_medians, "w", "Winner-Take-All Exponent (w)"),
        (delta_buckets, delta_medians, "δ", "Public Good Parameter (δ)"),
        (rho_buckets, rho_medians, "ρ", "Alignment Correlation (ρ)"),
        (z_buckets, z_medians, "z", "Misaligned AI Power Advantage (z)"),
        (china_buckets, china_medians, "China R", "China's Resource Share"),
        (amity_buckets, amity_medians, "Avg amity", "Average Amity (off-diagonal)"),
    ]

    return {
        "params": params_list,
        "human_shares": human_shares,
        "variables": variables,
    }


def compute_median_by_buckets(
    param_values: list[float], outcomes: list[float], n_buckets: int = 10
) -> tuple[list[float], list[float]]:
    """Compute conditional median outcome in each parameter decile.

    Returns (bucket_centers, bucket_medians) for n_buckets equal-percentile
    bins across the parameter range.
    """
    param_array = np.array(param_values)
    outcomes_array = np.array(outcomes)

    # Compute percentile boundaries for equal-sized buckets
    percentiles = np.linspace(0, 100, n_buckets + 1)
    boundaries = np.percentile(param_array, percentiles)

    bucket_centers = []
    bucket_medians = []

    for i in range(n_buckets):
        lower = boundaries[i]
        upper = boundaries[i + 1]

        # Handle floating point edge cases
        if i == 0:
            mask = param_array <= upper
        elif i == n_buckets - 1:
            mask = param_array >= lower
        else:
            mask = (param_array >= lower) & (param_array <= upper)

        if mask.sum() > 0:
            median_outcome = float(np.median(outcomes_array[mask]))
            center = float((lower + upper) / 2.0)
            bucket_centers.append(center)
            bucket_medians.append(median_outcome)

    return bucket_centers, bucket_medians
