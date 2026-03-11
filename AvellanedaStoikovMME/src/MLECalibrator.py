import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MLECalibrator:
    """
    Rolling-window MLE for the Avellaneda-Stoikov fill-intensity model:

        λ(δ) = A · exp(-k · δ)

    Parameters are estimated in log-space so the optimiser never proposes
    negative A or k.
    """

    def __init__(self, window_size: int = 5_000, halflife: float = 2_500):
        self.window_size = window_size
        self.halflife = halflife
        # Each row: [δ (spread distance), filled (0/1), dt (time interval)]
        self.data: list = []

    # ------------------------------------------------------------------
    def record_observation(self, delta: float, filled: int, dt: float):
        self.data.append([delta, filled, dt])
        if len(self.data) > self.window_size:
            self.data.pop(0)

    # ------------------------------------------------------------------
    def _weights(self, n: int) -> np.ndarray:
        """Exponential decay: most-recent observation has weight ≈ 1."""
        return np.exp(-np.log(2) / self.halflife * np.arange(n)[::-1])

    # ------------------------------------------------------------------
    def objective(self, params: np.ndarray) -> float:
        """
        Weighted Poisson log-likelihood (negative, for minimisation).

        For a small interval dt the number of fills is Poisson(λ·dt).
        Because fills ∈ {0, 1} we use the simplified form:
            ℓ_i = w_i · [ y_i · log(λ_i) - λ_i · dt_i ]
        """
        log_A, log_k = params
        A, k = np.exp(log_A), np.exp(log_k)

        arr = np.array(self.data)
        deltas = arr[:, 0]
        fills = arr[:, 1]
        dts = arr[:, 2]

        n = len(fills)
        weights = self._weights(n)

        lambdas = A * np.exp(-k * deltas)
        log_lambdas = np.log(np.clip(lambdas, 1e-12, None))

        log_lik = weights * (fills * log_lambdas - lambdas * dts)
        return -np.sum(log_lik)

    # ------------------------------------------------------------------
    def calibrate(self) -> np.ndarray | None:
        """
        Returns [A, k] on success, None if insufficient data or optimiser
        failed.

        Bounds (log-space):
            A ∈ [exp(0), exp(7)]   → [1, ~1097]
            k ∈ [exp(-2.3), exp(3)] → [0.1, ~20]
        """
        if len(self.data) < 100:
            return None

        bounds = [(0.0, 7.0), (-2.3, 3.0)]
        initial_guess = np.log([140.0, 1.5])

        res = minimize(
            self.objective,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 500},
        )

        if not res.success:
            # Retry from multiple starts before giving up
            for A0, k0 in [(50, 1.0), (200, 2.0), (100, 0.5)]:
                res = minimize(
                    self.objective,
                    np.log([A0, k0]),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 500},
                )
                if res.success:
                    break
            else:
                return None

        return np.exp(res.x)  # [A, k]

    # ------------------------------------------------------------------
    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data, columns=["delta", "filled", "dt"])
