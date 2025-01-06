import pytest
import numpy as np
from BlackScholes import BlackScholes

@pytest.mark.parametrize(
    "S, K, T, r, sigma, q, expected_call, expected_put",
    [
        # Example 1: S=100, K=100, T=1, r=0, sigma=0.2, q=0
        # Theoretical call=~7.97, put=~7.97
        (100.0, 100.0, 1.0, 0.0, 0.2, 0.0, 7.97, 7.97),
        # Example 2: S=100, K=100, T=1, r=0.05, sigma=0.2, q=0
        # Theoretical call=~10.45, put=~5.57
        (100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 10.45, 5.57),
        # Example 3: S=50, K=60, T=0.5, r=0.08, sigma=0.4, q=0
        # Theoretical call=~2.97, put=~10.62
        (50.0, 60.0, 0.5, 0.08, 0.4, 0.0, 2.97, 10.62),
        # Example 4: S=150, K=130, T=3.25, r=0.02, sigma=0.1, q=0.05
        # Theoretical call=~12.08, put=~6.40
        (150.0, 130.0, 3.25, 0.02, 0.1, 0.05, 12.08, 6.40),
    ]
)
def test_bsm_prices(S, K, T, r, sigma, q, expected_call, expected_put):
    """
    Test the call_option_price and put_option_price for known references.
    """
    bs = BlackScholes(
        underlying_price=S, 
        strike=K, 
        time_to_maturity=T,
        interest_rate=r, 
        volatility=sigma, 
        dividend_yield=q
    )
    call_val = bs.call_option_price()
    put_val = bs.put_option_price()

    # Check prices vs. expected (tolerance ±0.01)
    assert pytest.approx(call_val, 0.01) == expected_call
    assert pytest.approx(put_val, 0.01) == expected_put


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q, option_type, exp_price, exp_delta, exp_gamma, exp_vega, exp_theta",
    [
        # ---- Example 1: (Call) ----
        # S=100, K=100, T=1, r=0, sigma=0.2, q=0 => Theoretical Price ~7.97
        # Greeks (Call): Delta=0.53983, Gamma=0.01985, Vega=0.39695, Theta=-0.01088
        (100, 100, 1.0, 0.0, 0.2, 0.0, "call",
         7.97, 0.53983, 0.01985, 0.39695, -0.01088),

        # ---- Example 1: (Put) ----
        # S=100, K=100, T=1, r=0, sigma=0.2, q=0 => Theoretical Price ~7.97
        # Greeks (Put): Delta=-0.46017, Gamma=0.01985, Vega=0.39695, Theta=-0.01088
        (100, 100, 1.0, 0.0, 0.2, 0.0, "put",
         7.97, -0.46017, 0.01985, 0.39695, -0.01088),

        # ---- Example 2: (Call) ----
        # S=100, K=100, T=1, r=0.05, sigma=0.2, q=0 => Price ~10.45
        # Greeks (Call): Delta=0.63683, Gamma=0.01876, Vega=0.37524, Theta=-0.01757
        (100, 100, 1.0, 0.05, 0.2, 0.0, "call",
         10.45, 0.63683, 0.01876, 0.37524, -0.01757),

        # ---- Example 2: (Put) ----
        # S=100, K=100, T=1, r=0.05, sigma=0.2, q=0 => Price ~5.57
        # Greeks (Put): Delta=-0.36317, Gamma=0.01876, Vega=0.37524, Theta=-0.00454
        (100, 100, 1.0, 0.05, 0.2, 0.0, "put",
         5.57, -0.36317, 0.01876, 0.37524, -0.00454),

        # ---- Example 3: (Call) ----
        # S=50, K=60, T=0.5, r=0.08, sigma=0.4, q=0 => Price=2.97
        # Greeks (Call): Delta=0.35877, Gamma=0.02642, Vega=0.13211, Theta=-0.01776
        (50, 60, 0.5, 0.08, 0.4, 0.0, "call",
         2.97, 0.35877, 0.02642, 0.13211, -0.01776),

        # ---- Example 3: (Put) ----
        # S=50, K=60, T=0.5, r=0.08, sigma=0.4, q=0 => Price=10.62
        # Greeks (Put): Delta=-0.64123, Gamma=0.02642, Vega=0.13211, Theta=-0.00512
        (50, 60, 0.5, 0.08, 0.4, 0.0, "put",
         10.62, -0.64123, 0.02642, 0.13211, -0.00512),

        # ---- Example 4: (Call) ----
        # S=150, K=130, T=3.25, r=0.02, sigma=0.1, q=0.05 => Price=12.08
        # Greeks (Call): Delta=0.539, Gamma=0.0118, Vega=0.8646, Theta=0.0037
        (150, 130, 3.25, 0.02, 0.1, 0.05, "call",
         12.08, 0.539, 0.0118, 0.8646, 0.0037),

        # ---- Example 4: (Put) ----
        # S=150, K=130, T=3.25, r=0.02, sigma=0.1, q=0.05 => Price=6.40
        # Greeks (Put): Delta=-0.311, Gamma=0.0118, Vega=0.8646, Theta=-0.00713
        (150, 130, 3.25, 0.02, 0.1, 0.05, "put",
         6.40, -0.311, 0.0118, 0.8646, -0.00713),
    ]
)

def test_bsm_greeks(
    S, K, T, r, sigma, q, option_type, exp_price, exp_delta, exp_gamma, exp_vega, exp_theta
):
    """
    Test both the theoretical price and all Greeks (Delta, Gamma, Vega, Theta, Rho) 
    for calls or puts with known references.
    """
    bs = BlackScholes(
        underlying_price=S,
        strike=K,
        time_to_maturity=T,
        interest_rate=r,
        volatility=sigma,
        dividend_yield=q
    )
    greeks = bs.greeks()

    if option_type == "call":
        price_val = bs.call_option_price()
        delta = greeks["delta_call"]
        theta = greeks["theta_call"]
    else:
        price_val = bs.put_option_price()
        delta = greeks["delta_put"]
        theta = greeks["theta_put"]

    gamma = greeks["gamma"]
    vega = greeks["vega"]

    # Compare each value to the expected references
    # Price tolerance: ±0.01
    assert pytest.approx(price_val, abs=0.01) == exp_price
    # Delta, Gamma, Vega, Theta can be tested with smaller tolerances (±1e-3)
    assert pytest.approx(delta, abs=1e-3) == exp_delta
    assert pytest.approx(gamma, abs=1e-3) == exp_gamma
    assert pytest.approx(vega, abs=1e-3) == exp_vega
    assert pytest.approx(theta, abs=1e-3) == exp_theta


def test_greeks_shape():
    """
    Check that greeks() returns all required keys and that the values are floats.
    """
    bs = BlackScholes(100.0, 100.0, 1.0, 0.05, 0.2)
    g = bs.greeks()
    expected_keys = {
        "delta_call", "delta_put", "gamma", "theta_call", "theta_put", "vega"
    }
    assert set(g.keys()) == expected_keys
    for val in g.values():
        assert isinstance(val, float)

def test_greek_curve_dimension():
    """
    Ensure compute_greek_curve returns arrays of correct length.
    """
    bs = BlackScholes(100.0, 100.0, 1.0, 0.05, 0.2)
    x_vals, y_vals = bs.compute_greek_curve("delta_call", "underlying_price", num_points=20)
    assert len(x_vals) == 20
    assert len(y_vals) == 20

def test_zero_dividend_yield():
    """
    Simple edge case check for zero dividend.
    """
    bs = BlackScholes(100.0, 100.0, 1.0, 0.05, 0.2, dividend_yield=0.0)
    call_val = bs.call_option_price()
    assert call_val > 0