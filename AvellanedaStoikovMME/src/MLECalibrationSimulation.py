import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.MLECalibrator import MLECalibrator
from src.RegimeSimulator import RegimeSimulator


def run_calibration_tracking_experiment(
    n_steps: int = 4_000,
    calibrate_every: int = 200,
    window_size: int = 2_000,
    halflife: float = 1_000,
) -> tuple:
    """
    Runs the simulation and periodically re-calibrates the MLE model.

    Returns
    -------
    calibrator      : MLECalibrator  rolling window (tail = post-shock data)
    post_calibrator : MLECalibrator  pure post-shock observations only;
                      used for diagnostic plots so empirical statistics are
                      drawn from the same single-regime distribution that
                      the MLE was effectively fitted to
    history         : dict  keys: step, A_est, k_est
    sim             : RegimeSimulator
    """
    sim = RegimeSimulator()
    calibrator = MLECalibrator(window_size=window_size, halflife=halflife)
    post_calibrator = MLECalibrator(window_size=window_size, halflife=halflife)
    history = {"step": [], "A_est": [], "k_est": []}

    for step, delta, filled, dt in sim.generate(n_steps):
        calibrator.record_observation(delta, filled, dt)

        # Separate pure post-shock observations for clean diagnostic plots
        if step >= sim.shock_step:
            post_calibrator.record_observation(delta, filled, dt)

        if step > 0 and step % calibrate_every == 0:
            result = calibrator.calibrate()
            if result is not None:
                A_est, k_est = result
                history["step"].append(step)
                history["A_est"].append(A_est)
                history["k_est"].append(k_est)

    return calibrator, post_calibrator, history, sim


def compute_empirical_fill_prob(
    df: pd.DataFrame,
    n_bins: int = 12,
    min_count: int = 30,
    use_quantile_bins: bool = True,
) -> tuple:
    """
    Compute empirical fill probability per δ-bin.

    Parameters
    ----------
    n_bins            : number of bins
    min_count         : discard bins with fewer observations (sparse bins
                        produce unstable probability estimates)
    use_quantile_bins : if True, use quantile-based bins so each bin has
                        roughly equal sample count; if False, use equal-width
                        bins (original behaviour)
    """
    df = df.copy()

    if use_quantile_bins:
        # Quantile-based bins → equal occupancy, avoids sparse tail bins
        df["delta_bin"] = pd.qcut(df["delta"], q=n_bins, duplicates="drop")
    else:
        df["delta_bin"] = pd.cut(df["delta"], bins=n_bins)

    grouped = df.groupby("delta_bin", observed=True)
    empirical = grouped["filled"].mean()
    delta_mid = grouped["delta"].mean()
    counts = grouped["filled"].count()

    # Keep only bins with enough observations AND a non-NaN mean
    mask = (~np.isnan(empirical.values)) & (counts.values >= min_count)
    return delta_mid.values[mask], empirical.values[mask]


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------


def plot_learning_dynamics(history: dict, sim: RegimeSimulator):
    """Tracks A and k estimates over time, with regime-change marker."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(
        "MLE Learning Dynamics — Avellaneda-Stoikov Calibrator",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    steps = history["step"]
    A_ests = history["A_est"]
    k_ests = history["k_est"]

    # -- Panel A: arrival intensity
    ax = axes[0]
    ax.plot(steps, A_ests, color="#2563EB", lw=2, marker="o", ms=4, label="Estimated A")
    ax.axhline(
        sim.true_A,
        color="#DC2626",
        ls="--",
        lw=1.5,
        label=f"True A (pre-shock) = {sim.true_A}",
    )
    ax.axhline(
        sim.shock_A,
        color="#F97316",
        ls=":",
        lw=1.5,
        label=f"True A (post-shock) = {sim.shock_A}",
    )
    ax.axvline(
        sim.shock_step,
        color="gray",
        ls="-.",
        lw=1.2,
        label=f"Regime shift @ t={sim.shock_step}",
    )
    ax.set_ylabel("Arrival Intensity (A)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.35)

    # -- Panel k: liquidity decay
    ax = axes[1]
    ax.plot(steps, k_ests, color="#16A34A", lw=2, marker="o", ms=4, label="Estimated k")
    ax.axhline(
        sim.true_k,
        color="#DC2626",
        ls="--",
        lw=1.5,
        label=f"True k (pre-shock) = {sim.true_k}",
    )
    ax.axhline(
        sim.shock_k,
        color="#F97316",
        ls=":",
        lw=1.5,
        label=f"True k (post-shock) = {sim.shock_k}",
    )
    ax.axvline(sim.shock_step, color="gray", ls="-.", lw=1.2)
    ax.set_ylabel("Liquidity Decay (k)", fontsize=11)
    ax.set_xlabel("Observation Step", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------


def plot_mle_fit(
    calibrator: MLECalibrator, post_calibrator: MLECalibrator, sim: RegimeSimulator
):
    """
    Fill-probability fit: empirical scatter vs MLE Poisson-CDF curve.

    Empirical points are drawn from `post_calibrator` (pure post-shock data)
    so they reflect a single homogeneous regime.  The MLE parameters come
    from `calibrator` (the rolling-window estimator), which by the end of the
    run is also dominated by post-shock observations.

    The x-axis is capped at the 95th-percentile of δ in the post-shock window
    so the near-zero tail does not compress the informative region.
    """
    df_post = post_calibrator.as_dataframe()
    if df_post.empty:
        print("No post-shock calibration data available yet.")
        return

    result = calibrator.calibrate()
    if result is None:
        print("Not enough data for calibration.")
        return

    A, k = result
    dt_mean = df_post["dt"].mean()

    print(f"\n{'='*48}")
    print(f"  MLE Calibration Results  (post-shock regime)")
    print(f"{'='*48}")
    print(f"  Estimated A  : {A:.4f}   (true = {sim.shock_A})")
    print(f"  Estimated k  : {k:.4f}   (true = {sim.shock_k})")
    print(f"  Post-shock fill rate : {df_post['filled'].mean():.4f}")
    print(
        f"  δ range              : [{df_post['delta'].min():.3f}, "
        f"{df_post['delta'].max():.3f}]"
    )
    print(f"{'='*48}\n")

    # Cap x-axis: beyond the 95th-pct the probability is so small that
    # empirical bins are unreliable and the curve is indistinguishable from 0
    delta_cap = float(np.percentile(df_post["delta"], 95))

    n_post = len(df_post)
    n_bins_fit = min(18, max(10, n_post // 400))
    min_cnt_fit = max(40, n_post // 150)

    delta_emp, prob_emp = compute_empirical_fill_prob(
        df_post[df_post["delta"] <= delta_cap],
        n_bins=n_bins_fit,
        min_count=min_cnt_fit,
        use_quantile_bins=True,
    )
    delta_grid = np.linspace(df_post["delta"].min(), delta_cap, 300)

    lam_grid = A * np.exp(-k * delta_grid)
    model_prob = 1.0 - np.exp(-lam_grid * dt_mean)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        delta_emp,
        prob_emp,
        s=70,
        zorder=5,
        color="#2563EB",
        label="Empirical fill prob (post-shock)",
    )
    ax.plot(
        delta_grid,
        model_prob,
        lw=2.5,
        color="#DC2626",
        label=f"MLE fit  A={A:.2f}, k={k:.2f}",
    )

    # Overlay true model curve for comparison
    lam_true = sim.shock_A * np.exp(-sim.shock_k * delta_grid)
    prob_true = 1.0 - np.exp(-lam_true * dt_mean)
    ax.plot(
        delta_grid,
        prob_true,
        lw=1.8,
        color="#16A34A",
        ls=":",
        label=f"True model  A={sim.shock_A}, k={sim.shock_k}",
    )

    ax.set_xlabel("Spread Distance (δ)", fontsize=12)
    ax.set_ylabel("Fill Probability", fontsize=12)
    ax.set_title(
        "MLE Calibration of Fill Intensity  λ(δ) = A·exp(−k·δ)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------


def plot_intensity_diagnostic(
    calibrator: MLECalibrator, post_calibrator: MLECalibrator, sim: RegimeSimulator
):
    """
    Research diagnostic: log λ̂(δ) vs δ should be linear under the AS model:
        log λ = log A  −  k · δ

    Design choices
    --------------
    * Equal-width δ bins (NOT quantile) so the x-axis is uniform and the
      slope is visually correct.  Quantile bins over-crowd the low-δ region
      where fills are common and compress the high-δ tail — exactly backwards
      for a slope diagnostic.
    * δ cap driven by raw fill count, not fill probability: keep every bin
      with ≥ MIN_FILLS actual fills.  This is the correct criterion because
      the Poisson CDF inversion only has acceptable variance when the sample
      fill count is large enough.
    * Error bars via Wilson binomial CI propagated through the log-λ transform,
      so the viewer can see which points are noisy vs well-estimated.
    """
    df_post = post_calibrator.as_dataframe()
    if df_post.empty:
        print("No post-shock data.")
        return

    result = calibrator.calibrate()
    if result is None:
        print("Not enough data for calibration.")
        return

    A_mle, k_mle = result
    dt_mean = df_post["dt"].mean()
    n_post = len(df_post)

    # ------------------------------------------------------------------
    # 1. Equal-width bins across the full δ range
    # ------------------------------------------------------------------
    N_BINS = 14  # 14 bins × width≈0.107 over [0.05,1.5]; ~571 obs each
    MIN_FILLS = 5  # Wilson CI still valid at n≥5; captures out to δ≈1.4
    delta_min = df_post["delta"].min()
    delta_max = df_post["delta"].max()
    edges = np.linspace(delta_min, delta_max, N_BINS + 1)
    bin_mids = 0.5 * (edges[:-1] + edges[1:])

    deltas = df_post["delta"].values
    fills = df_post["filled"].values

    bin_idx = np.digitize(deltas, edges) - 1
    bin_idx = np.clip(bin_idx, 0, N_BINS - 1)

    bin_n = np.zeros(N_BINS, dtype=int)
    bin_fills = np.zeros(N_BINS, dtype=int)
    for i, f in zip(bin_idx, fills):
        bin_n[i] += 1
        bin_fills[i] += int(f)

    # ------------------------------------------------------------------
    # 2. Keep only bins with enough fills (raw count, not probability)
    # ------------------------------------------------------------------
    valid = (bin_fills >= MIN_FILLS) & (bin_n > 0)

    d_v = bin_mids[valid]
    n_v = bin_n[valid]
    fills_v = bin_fills[valid]
    prob_v = fills_v / n_v  # empirical fill probability

    # ------------------------------------------------------------------
    # 3. CDF inversion: P = 1 - exp(-λ·dt)  =>  λ = -log(1-P)/dt
    # ------------------------------------------------------------------
    lam_v = -np.log(1.0 - np.clip(prob_v, 0.0, 1.0 - 1e-12)) / dt_mean
    log_lam = np.log(lam_v)

    # ------------------------------------------------------------------
    # 4. Propagated error bars via Wilson CI on the fill probability,
    #    then delta-method through the log-λ transform
    # ------------------------------------------------------------------
    z = 1.96  # 95 % CI
    # Wilson interval for proportion
    p_lo = (
        fills_v + 0.5 * z**2 - z * np.sqrt(fills_v * (1 - prob_v) + 0.25 * z**2)
    ) / (n_v + z**2)
    p_hi = (
        fills_v + 0.5 * z**2 + z * np.sqrt(fills_v * (1 - prob_v) + 0.25 * z**2)
    ) / (n_v + z**2)
    p_lo = np.clip(p_lo, 1e-9, 1.0 - 1e-12)
    p_hi = np.clip(p_hi, 1e-9, 1.0 - 1e-12)

    lam_lo = -np.log(1.0 - p_lo) / dt_mean
    lam_hi = -np.log(1.0 - p_hi) / dt_mean
    err_lo = log_lam - np.log(lam_lo)  # downward error bar in log space
    err_hi = np.log(lam_hi) - log_lam  # upward error bar in log space

    # ------------------------------------------------------------------
    # 5. Reference lines — WLS slope weighted by inverse log-λ variance
    # ------------------------------------------------------------------
    # Variance of log λ̂ via delta method:
    #   Var(log λ̂) ≈ Var(P̂) / [P(1-P) * (λ·dt)²]
    # For a binomial proportion:  Var(P̂) = P(1-P)/n
    # => Var(log λ̂) ≈ 1 / [n * P * (1-P) * (λ·dt)²]  ×  (1/(1-P))²
    # In practice we use the CI half-width as a proxy: w_i = 1/(err_lo + err_hi)²
    # This ensures tight (high-fill) bins dominate the slope estimate.
    ci_width = err_lo + err_hi + 1e-9  # total CI width in log space
    wls_w = 1.0 / ci_width**2  # inverse-variance weights

    # Weighted least squares via np.polyfit with w parameter
    wls_coef = np.polyfit(d_v, log_lam, 1, w=wls_w)
    wls_k = -wls_coef[0]

    # Also keep unweighted OLS for comparison
    ols_coef = np.polyfit(d_v, log_lam, 1)
    ols_k = -ols_coef[0]

    d_lo_line = d_v.min()
    d_hi_line = d_v.max()
    delta_line = np.linspace(d_lo_line, d_hi_line, 300)
    log_lam_mle = np.log(A_mle) - k_mle * delta_line
    log_lam_true = np.log(sim.shock_A) - sim.shock_k * delta_line
    wls_line = np.polyval(wls_coef, delta_line)

    # Reliability score per bin (0→1): drives marker size & alpha
    reliability = wls_w / wls_w.max()
    marker_sizes = 40 + 80 * reliability  # large=reliable, small=noisy
    marker_alphas = 0.4 + 0.6 * reliability  # opaque=reliable, faded=noisy

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    # Draw reference lines first (behind points)
    ax.plot(
        delta_line,
        log_lam_mle,
        color="#DC2626",
        lw=2.2,
        ls="--",
        label=f"MLE:  log A={np.log(A_mle):.2f}, k={k_mle:.2f}",
    )
    ax.plot(
        delta_line,
        log_lam_true,
        color="#16A34A",
        lw=1.8,
        ls=":",
        label=f"True: log A={np.log(sim.shock_A):.2f}, k={sim.shock_k:.2f}",
    )
    ax.plot(
        delta_line,
        wls_line,
        color="#6366F1",
        lw=1.8,
        ls="-.",
        label=f"WLS fit (inv-var weighted, slope = -{wls_k:.2f})",
    )

    # Error bars with per-point alpha encoding reliability
    for i in range(len(d_v)):
        alpha_i = float(marker_alphas[i])
        ax.errorbar(
            d_v[i : i + 1],
            log_lam[i : i + 1],
            yerr=[[err_lo[i]], [err_hi[i]]],
            fmt="o",
            color="#7C3AED",
            ms=float(marker_sizes[i]) ** 0.5,
            alpha=alpha_i,
            lw=1.0,
            capsize=3,
            capthick=1.0,
            zorder=5,
        )

    # Single legend entry for all empirical points
    ax.scatter(
        [],
        [],
        marker="o",
        color="#7C3AED",
        s=60,
        label=f"Empirical log λ ± 95% CI  ({valid.sum()} bins, "
        f"≥{MIN_FILLS} fills; size ∝ reliability)",
    )

    ax.set_xlabel("Spread Distance (δ)", fontsize=12)
    ax.set_ylabel("log Fill Intensity  log λ", fontsize=12)
    ax.set_title(
        "Intensity Diagnostic — log λ vs δ  (linear => AS model valid)",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.97,
        0.97,
        f"WLS slope : -{wls_k:.3f}\n"
        f"OLS slope : -{ols_k:.3f}\n"
        f"MLE k     :  {k_mle:.3f}\n"
        f"True k    :  {sim.shock_k:.3f}\n"
        f"Post-shock n : {n_post:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.88),
    )
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(0.70, 0.99))
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()
    print(
        f"  Diagnostic: {valid.sum()} reliable bins | δ range [{d_v.min():.2f}, "
        f"{d_v.max():.2f}] | WLS k={wls_k:.3f} | OLS k={ols_k:.3f} | "
        f"MLE k={k_mle:.3f} | True k={sim.shock_k:.3f}"
    )
