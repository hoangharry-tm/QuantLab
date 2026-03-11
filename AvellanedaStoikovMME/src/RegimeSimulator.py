import numpy as np


class RegimeSimulator:
    """
    Synthetic market with a mid-session volatility / liquidity regime shift.
    Generates observations of the form (δ, filled, dt) that the calibrator
    can learn from, then injects a shock.

    Fill probability follows the AS model:
        P(fill | δ) = 1 - exp(-A · exp(-k · δ) · dt)
    """

    def __init__(
        self,
        true_A: float = 140.0,
        true_k: float = 1.5,
        shock_A: float = 60.0,  # post-shock arrival rate
        shock_k: float = 2.5,  # post-shock decay (tighter market)
        shock_step: int = 4_000,  # when the regime flips
        dt: float = 0.005,
        delta_range: tuple = (0.05, 2.5),
        rng_seed: int = 42,
    ):
        self.true_A = true_A
        self.true_k = true_k
        self.shock_A = shock_A
        self.shock_k = shock_k
        self.shock_step = shock_step
        self.dt = dt
        self.delta_range = delta_range
        self.rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    def _fill_prob(self, delta: float, A: float, k: float) -> float:
        lam = A * np.exp(-k * delta)
        return 1.0 - np.exp(-lam * self.dt)

    # ------------------------------------------------------------------
    def generate(self, n_steps: int = 4_000):
        """Yields (step, delta, filled, dt) tuples."""
        for step in range(n_steps):
            A = self.shock_A if step >= self.shock_step else self.true_A
            k = self.shock_k if step >= self.shock_step else self.true_k

            # Uniform sampling over the informationally-rich δ range.
            # The upper bound (1.5) is set so the tail bins still have
            # enough expected fills for reliable CDF inversion at post-shock
            # parameters (A≈60, k≈2.5): E[fills] ≈ n_per_bin * P(fill|δ=1.4)
            # ≈ 571 * 0.009 ≈ 5, which is sufficient for the Wilson CI.
            delta = self.rng.uniform(*self.delta_range)
            p_fill = self._fill_prob(delta, A, k)
            filled = int(self.rng.random() < p_fill)

            yield step, delta, filled, self.dt
