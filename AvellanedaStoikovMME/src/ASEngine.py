import numpy as np
import pandas as pd

class AvellanedaStoikovEngine:
    """
    Core implementation of the Avellaneda-Stoikov (2008) Market Making Model.
    Focuses on the calculation of the Reservation Price and the Optimal Spreads 
    using vectorized NumPy operations for high-performance benchmarks.
    """
    def __init__(self, gamma=0.01, k=1.5, A=140.0, sigma=2.0):
        self.gamma = gamma  # Risk aversion parameter
        self.k = k          # Order book depth (liquidity decay)
        self.A = A          # Arrival intensity of market orders
        self.sigma = sigma  # Price volatility (standard deviation)

    def calculate_reservation_price(self, s, q, t, T):
        """
        Calculates r(s, q, t): The price at which the MM is indifferent 
        to their current inventory risk.
        
        Equation: r = s - q * gamma * sigma^2 * (T - t)
        """
        time_remaining = T - t
        return s - (q * self.gamma * (self.sigma ** 2) * time_remaining)

    def calculate_optimal_spread(self, t, T):
        """
        Calculates the symmetric spread around the reservation price.
        
        Equation: delta_bid + delta_ask = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
        """
        time_remaining = T - t
        first_term = self.gamma * (self.sigma ** 2) * time_remaining
        second_term = (2 / self.gamma) * np.log(1 + (self.gamma / self.k))
        return first_term + second_term

    def compute_quotes(self, mid_prices, inventory, time_steps, total_time):
        """
        Vectorized computation of optimal Bid and Ask prices across a time series.
        
        Parameters:
        mid_prices (np.array): Market mid-prices (s)
        inventory (np.array): Current inventory levels (q)
        time_steps (np.array): Current time indices (t)
        total_time (float): Horizon (T)
        """
        # Ensure inputs are numpy arrays for vectorization
        s = np.array(mid_prices)
        q = np.array(inventory)
        t = np.array(time_steps)
        T = total_time
        
        # Calculate Reservation Price (r)
        res_prices = self.calculate_reservation_price(s, q, t, T)
        
        # Calculate Half-Spread (delta)
        total_spreads = self.calculate_optimal_spread(t, T)
        half_spreads = total_spreads / 2
        
        # Determine Final Quotes
        bid_quotes = res_prices - half_spreads
        ask_quotes = res_prices + half_spreads
        
        return pd.DataFrame({
            'mid_price': s,
            'reservation_price': res_prices,
            'optimal_bid': bid_quotes,
            'optimal_ask': ask_quotes,
            'spread': total_spreads
        })

    def get_fill_probabilities(self, delta):
        """
        Calculates the probability of an order being filled given its depth. It
        follows the Poisson distribution rate.
        
        Equation: Pr(fill) = A * exp(-k * delta)
        """
        return self.A * np.exp(-self.k * delta)
