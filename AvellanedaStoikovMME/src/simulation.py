import mplfinance as mpf
import numpy as np
import pandas as pd

from .ASEngine import AvellanedaStoikovEngine


def run_perfect_scenario_sim(
    S0=100, T=1, steps=1000, gamma=0.1, k=1.5, A=140, sigma=2.0
):
    """
    Simulates a 'Perfect Scenario' where mid-price is a pure random walk
    and the AS engine successfully captures the spread.
    """
    dt = T / steps
    engine = AvellanedaStoikovEngine(gamma=gamma, k=k, A=A, sigma=sigma)

    # 1. Generate Pure Random Walk (No Drift)
    shocks = np.random.normal(0, sigma * np.sqrt(dt), steps)
    prices = S0 + np.cumsum(shocks)

    # 2. Initialize State
    inventory = np.zeros(steps)
    cash = np.zeros(steps)
    pnl = np.zeros(steps)
    bids, asks = np.zeros(steps), np.zeros(steps)

    # 3. Execution Loop
    for t in range(1, steps):
        s = prices[t]
        q = inventory[t - 1]

        # Calculate optimal quotes for the current state
        # We normalize time to progress from 0 to T
        current_time = t * dt
        r = engine.calculate_reservation_price(s, q, current_time, T)
        spread = engine.calculate_optimal_spread(current_time, T)

        bid, ask = r - spread / 2, r + spread / 2
        bids[t], asks[t] = bid, ask

        # Calculate fill probabilities based on distance from mid
        prob_buy = engine.get_fill_probabilities(s - bid) * dt
        prob_sell = engine.get_fill_probabilities(ask - s) * dt

        # Simulate fills
        fill_buy = 1 if np.random.random() < prob_buy else 0
        fill_sell = 1 if np.random.random() < prob_sell else 0

        # Update Inventory and Cash
        inventory[t] = q + (fill_buy - fill_sell)
        cash[t] = cash[t - 1] + (fill_sell * ask) - (fill_buy * bid)

        # Mark-to-Market PnL
        pnl[t] = cash[t] + (inventory[t] * s)

    return pd.DataFrame(
        {
            "step": np.arange(steps),
            "mid": prices,
            "bid": bids,
            "ask": asks,
            "inventory": inventory,
            "pnl": pnl,
        }
    )


def simulate_toxic_flow(S0=100, T=5, steps=5000, drift=0.015):
    dt = T / steps
    engine = AvellanedaStoikovEngine(gamma=0.1, k=1.5, A=140, sigma=2.0)

    # Initialize simulation arrays and stochastic components
    shocks = np.random.normal(0, 4.0 * np.sqrt(dt), steps)
    prices = np.zeros(steps)
    prices[0] = S0
    inventory, cash, pnl = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    bids, asks, fills_sell = np.zeros(steps), np.zeros(steps), [np.nan] * steps

    for t in range(1, steps):
        shock = shocks[t]

        # Update mid-price based on current drift and random shock
        prices[t] = prices[t - 1] + shock + drift * dt

        # Calculate optimal quotes based on the Avellaneda-Stoikov framework
        s, q, curr_t = prices[t], inventory[t - 1], t * dt
        r = engine.calculate_reservation_price(s, q, curr_t, T)
        spread = engine.calculate_optimal_spread(curr_t, T)
        bid, ask = r - spread / 2, r + spread / 2

        bids[t] = bid
        asks[t] = ask

        # Determine fill probabilities with an asymmetric skew (Toxic Flow)
        prob_sell = min(1.0, engine.get_fill_probabilities(max(0, ask - s)) * 2.5 * dt)
        prob_buy = min(1.0, engine.get_fill_probabilities(max(0, s - bid)) * 0.7 * dt)

        # Execute trades using a single random draw for mutually exclusive fills
        u = np.random.random()
        fill_sell = int(u < prob_sell)
        fill_buy = int((u >= prob_sell) and (u < prob_sell + prob_buy))

        # MODELING ADVERSE SELECTION:
        # Trades cause immediate price impact and trend persistence
        price_impact = np.random.normal(3 * spread, spread)
        trend_strength = 0.2

        if fill_sell:
            fills_sell[t] = ask
            prices[t] = ask + price_impact  # Price moves against the MM after selling
            drift += trend_strength  # Informed buying creates upward momentum

        if fill_buy:
            prices[t] = bid - price_impact
            drift -= trend_strength

        drift = np.clip(drift, -1.0, 1.0)

        # Update accounting: Inventory, Cash, and Mark-to-Market PnL
        inventory[t] = q + (fill_buy - fill_sell)
        cash[t] = cash[t - 1] + (fill_sell * ask) - (fill_buy * bid)
        pnl[t] = cash[t] + inventory[t] * prices[t]

    print("Total fills:", np.sum(~np.isnan(fills_sell)))
    print("Final inventory:", inventory[-1])
    print("Final PnL:", pnl[-1])

    return pd.DataFrame(
        {
            "step": np.arange(steps),
            "mid": prices,
            "bid": bids,
            "ask": asks,
            "inventory": inventory,
            "pnl": pnl,
            "fills_sell": fills_sell,
        }
    )


def simulate_volatility_clustering(S0=100, T=5, steps=5000):
    dt = T / steps

    # AS engine still assumes LOW volatility
    engine = AvellanedaStoikovEngine(
        gamma=0.1, k=1.5, A=140, sigma=1.2  # stale volatility estimate
    )

    # True market volatility regimes
    sigma_low = 1.2
    sigma_high = 6.0

    crash_start = int(steps * 0.45)
    crash_end = int(steps * 0.75)

    prices = np.zeros(steps)
    prices[0] = S0

    inventory = np.zeros(steps)
    cash = np.zeros(steps)
    pnl = np.zeros(steps)

    bids = np.zeros(steps)
    asks = np.zeros(steps)

    fills_sell = [np.nan] * steps
    volatility = np.zeros(steps)

    drift = 0.0

    for t in range(1, steps):
        # -------- VOLATILITY REGIME --------
        if crash_start < t < crash_end:
            # persistent clustered volatility
            if np.random.random() < 0.85:
                sigma = sigma_high
            else:
                sigma = sigma_low
        else:
            sigma = sigma_low

        volatility[t] = sigma
        shock = np.random.normal(0, sigma * np.sqrt(dt))
        prices[t] = prices[t - 1] + shock + drift * dt

        # -------- AVELLANEDA-STOIKOV QUOTES --------
        s = prices[t]
        q = inventory[t - 1]
        curr_t = t * dt

        r = engine.calculate_reservation_price(s, q, curr_t, T)
        spread = engine.calculate_optimal_spread(curr_t, T)

        bid = r - spread / 2
        ask = r + spread / 2

        bids[t] = bid
        asks[t] = ask

        # -------- HIGH VOLATILITY = MORE FILLS --------
        vol_multiplier = 1 + (sigma / sigma_low)

        prob_sell = min(
            1.0, engine.get_fill_probabilities(max(0, ask - s)) * vol_multiplier * dt
        )
        prob_buy = min(
            1.0, engine.get_fill_probabilities(max(0, s - bid)) * vol_multiplier * dt
        )

        u = np.random.random()

        fill_sell = int(u < prob_sell)
        fill_buy = int((u >= prob_sell) and (u < prob_sell + prob_buy))

        # -------- ADVERSE PRICE MOVES --------
        price_impact = np.random.normal(
            4 * spread * vol_multiplier, spread * vol_multiplier
        )

        if fill_sell:
            fills_sell[t] = ask
            prices[t] = ask + price_impact
            drift += 0.4

        if fill_buy:
            prices[t] = bid - price_impact
            drift -= 0.4

        drift = np.clip(drift, -2.0, 2.0)

        # -------- ACCOUNTING --------
        inventory[t] = q + (fill_buy - fill_sell)
        cash[t] = cash[t - 1] + (fill_sell * ask) - (fill_buy * bid)
        pnl[t] = cash[t] + inventory[t] * prices[t]

    print("Total fills:", np.sum(~np.isnan(fills_sell)))
    print("Final inventory:", inventory[-1])
    print("Final PnL:", pnl[-1])

    return pd.DataFrame(
        {
            "step": np.arange(steps),
            "mid": prices,
            "bid": bids,
            "ask": asks,
            "inventory": inventory,
            "pnl": pnl,
            "fills_sell": fills_sell,
            "volatility": volatility,
        }
    )


def simulate_execution_latency(S0=100, T=5, steps=5000, latency_steps=20):
    dt = T / steps

    engine = AvellanedaStoikovEngine(gamma=0.1, k=1.5, A=140, sigma=2.0)

    prices = np.zeros(steps)
    prices[0] = S0

    inventory = np.zeros(steps)
    cash = np.zeros(steps)
    pnl = np.zeros(steps)

    bids = np.zeros(steps)
    asks = np.zeros(steps)

    perceived_mid = np.zeros(steps)
    fills_sell = [np.nan] * steps
    fills_buy = [np.nan] * steps

    volatility = 2.0
    drift = 0

    # flash crash window
    crash_start = int(steps * 0.45)
    crash_end = int(steps * 0.65)

    # --- Latency state ---
    last_update_price = S0
    last_update_step = 0

    for t in range(1, steps):

        # -----------------------------------
        # 1. TRUE MARKET EVOLUTION
        # -----------------------------------

        # volatility spike during crash
        sigma = volatility * 4 if crash_start < t < crash_end else volatility

        shock = np.random.normal(0, sigma * np.sqrt(dt))

        prices[t] = prices[t - 1] + shock + drift * dt

        # occasional information jump (another exchange)
        if np.random.random() < 0.002:
            jump = np.random.normal(0, 8 * sigma)
            prices[t] += jump

        # -----------------------------------
        # 2. LATENCY (MM sees old price)
        # -----------------------------------

        # delayed_idx = max(0, t - latency_steps)
        # s_perceived = prices[delayed_idx]
        # perceived_mid[t] = s_perceived

        if t - last_update_step >= latency_steps:
            last_update_price = prices[t]
            last_update_step = t

        s_perceived = last_update_price
        perceived_mid[t] = s_perceived

        # -----------------------------------
        # 3. MM QUOTES (based on stale data)
        # -----------------------------------

        q = inventory[t - 1]
        curr_t = t * dt

        r = engine.calculate_reservation_price(s_perceived, q, curr_t, T)

        spread = engine.calculate_optimal_spread(curr_t, T)

        bid = r - spread / 2
        ask = r + spread / 2

        bids[t] = bid
        asks[t] = ask

        # -----------------------------------
        # 4. PREDATORY PICK-OFF LOGIC
        # -----------------------------------

        fill_buy = 0
        fill_sell = 0

        # If true price crosses stale quotes → guaranteed fill
        if prices[t] > ask:
            fill_sell = 1
            fills_sell[t] = ask

        elif prices[t] < bid:
            fill_buy = 1
            fills_buy[t] = bid

        else:

            # otherwise small passive fill chance
            prob_sell = engine.get_fill_probabilities(max(0, ask - prices[t])) * dt

            prob_buy = engine.get_fill_probabilities(max(0, prices[t] - bid)) * dt

            u = np.random.random()

            fill_sell = int(u < prob_sell)
            fill_buy = int((u >= prob_sell) and (u < prob_sell + prob_buy))

        # -----------------------------------
        # 5. ADVERSE SELECTION
        # -----------------------------------

        # price continues in same direction
        continuation = np.random.normal(2 * spread, spread)

        if fill_sell:

            prices[t] += continuation
            drift += 0.5

        if fill_buy:

            prices[t] -= continuation
            drift -= 0.5

        drift = np.clip(drift, -3, 3)

        # -----------------------------------
        # 6. ACCOUNTING
        # -----------------------------------

        inventory[t] = q + (fill_buy - fill_sell)

        cash[t] = cash[t - 1] + fill_sell * ask - fill_buy * bid

        pnl[t] = cash[t] + inventory[t] * prices[t]

    print("Final inventory:", inventory[-1])
    print("Final PnL:", pnl[-1])

    return pd.DataFrame(
        {
            "step": np.arange(steps),
            "mid": prices,
            "perceived": perceived_mid,
            "bid": bids,
            "ask": asks,
            "inventory": inventory,
            "pnl": pnl,
            "fills_sell": fills_sell,
            "fills_buy": fills_buy,
        }
    )
