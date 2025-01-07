import pytest
import numpy as np
from BlackScholes import BlackScholes
from US_YieldCurve import fetch_us_yield_curve_with_maturities
from streamlit_app import calculate_risk_free_rate_from_yield_curve
import requests
from unittest.mock import patch, MagicMock
from math import isclose


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q, expected_call, expected_put",
    [
        # Example 1: S=100, K=100, T=1, r=0, sigma=0.2 => Theoretical call=~7.97, put=~7.97
        (100.0, 100.0, 1.0, 0.0, 0.2, 0.0, 7.97, 7.97),
        # Example 2: S=100, K=100, T=1, r=0.05, sigma=0.2 => call=~10.45, put=~5.57
        (100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 10.45, 5.57),
        # Example 3: S=50, K=60, T=0.5, r=0.08, sigma=0.4 => call=~2.97, put=~10.62
        (50.0, 60.0, 0.5, 0.08, 0.4, 0.0, 2.97, 10.62),
        # Example 4: S=150, K=130, T=3.25, r=0.02, sigma=0.1, q=0.05 => call=~12.08, put=~6.40
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

    assert pytest.approx(call_val, abs=0.01) == expected_call
    assert pytest.approx(put_val, abs=0.01) == expected_put


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q, option_type, exp_price, exp_delta, exp_gamma,"
    "exp_vega, exp_theta",
    [
        # (Call) S=100, K=100, T=1, r=0, sigma=0.2 => Price~7.97
        # Greeks (Call): Delta=0.5398, Gamma=0.01985, Vega=0.39695, Theta=~ -0.01088
        (100, 100, 1.0, 0.0, 0.2, 0.0, "call",
         7.97, 0.5398, 0.01985, 0.39695, -0.01088),

        # (Put) Same scenario => Price~7.97
        # Delta=-0.4602, Gamma=0.01985, Vega=0.39695, Theta=~ -0.01088
        (100, 100, 1.0, 0.0, 0.2, 0.0, "put",
         7.97, -0.4602, 0.01985, 0.39695, -0.01088),

        # (Call) S=100, K=100, T=1, r=0.05 => Price=~10.45
        # Greeks: Delta=0.6368, Gamma=0.01876, Vega=0.37524, Theta=~ -0.01757
        (100, 100, 1.0, 0.05, 0.2, 0.0, "call",
         10.45, 0.6368, 0.01876, 0.37524, -0.01757),

        # (Put) => Price=~5.57, Delta=-0.3632, Gamma=0.01876, Vega=0.37524, Theta=~ -0.00454
        (100, 100, 1.0, 0.05, 0.2, 0.0, "put",
         5.57, -0.3632, 0.01876, 0.37524, -0.00454),
    ]
)
def test_bsm_greeks(
    S, K, T, r, sigma, q, option_type, exp_price,
    exp_delta, exp_gamma, exp_vega, exp_theta
):
    """
    Test both the theoretical price and Greeks (Delta, Gamma, Vega, Theta)
    for calls or puts with known references.
    """
    bs = BlackScholes(S, K, T, r, sigma, q)
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

    # Tolerances
    assert pytest.approx(price_val, abs=0.01) == exp_price
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


@pytest.mark.parametrize("param_name", [
    "underlying_price", "strike", "time_to_maturity", "interest_rate", "volatility"
])
def test_greek_curve_valid_params(param_name):
    """
    Check compute_greek_curve does not raise for valid param_name.
    """
    bs = BlackScholes(100.0, 100.0, 1.0, 0.05, 0.2)
    x_vals, y_vals = bs.compute_greek_curve("delta_call", param_name)
    assert len(x_vals) > 0
    assert len(y_vals) > 0


def test_greek_curve_invalid_param():
    """
    Check compute_greek_curve raises ValueError for invalid param_name.
    """
    bs = BlackScholes(100.0, 100.0, 1.0, 0.05, 0.2)
    with pytest.raises(ValueError):
        bs.compute_greek_curve("delta_call", "invalid_param")


def test_greek_curve_invalid_greek():
    """
    Check compute_greek_curve raises ValueError for invalid greek_name.
    """
    bs = BlackScholes(100.0, 100.0, 1.0, 0.05, 0.2)
    with pytest.raises(ValueError):
        bs.compute_greek_curve("invalid_greek", "underlying_price")


@pytest.mark.parametrize(
    "param,val",
    [
        ("underlying_price", -1.0),
        ("strike", -5.0),
        ("time_to_maturity", -0.5),
        ("interest_rate", -0.02),
        ("volatility", -0.3),
    ]
)
def test_black_scholes_negative_params(param, val):
    """
    Check that BlackScholes constructor raises ValueError
    if a key parameter is negative.
    """
    kwargs = {
        "underlying_price": 100.0,
        "strike": 100.0,
        "time_to_maturity": 1.0,
        "interest_rate": 0.05,
        "volatility": 0.2
    }
    kwargs[param] = val
    with pytest.raises(ValueError):
        BlackScholes(**kwargs)


def test_black_scholes_negative_dividend():
    """
    Check that negative dividend_yield raises ValueError.
    """
    with pytest.raises(ValueError):
        BlackScholes(100.0, 100.0, 1.0, 0.05, 0.2, dividend_yield=-0.01)


def test_black_scholes_zero_vol_or_zero_time():
    """
    Edge case: if volatility=0 or time_to_maturity=0, check greeks are zero and d1,d2=0.
    """
    bs = BlackScholes(100, 100, 0, 0.05, 0.2)  # T=0
    assert bs.calculate_d1_d2() == (0.0, 0.0)
    g = bs.greeks()
    for val in g.values():
        assert val == 0.0

    bs2 = BlackScholes(100, 100, 1.0, 0.05, 0.0)  # sigma=0
    assert bs2.calculate_d1_d2() == (0.0, 0.0)
    g2 = bs2.greeks()
    for val in g2.values():
        assert val == 0.0


def test_generate_heatmaps_invalid_ranges():
    """
    Check that generate_heatmaps raises ValueError if the spot or volatility range
    starts at or below 0.
    """
    bs = BlackScholes(100, 100, 1, 0.05, 0.2)
    with pytest.raises(ValueError):
        bs.generate_heatmaps((0, 120), (0.1, 0.5))
    with pytest.raises(ValueError):
        bs.generate_heatmaps((100, 120), (0, 0.5))


def test_zero_dividend_yield():
    """
    Simple edge case check for zero dividend.
    """
    bs = BlackScholes(100.0, 100.0, 1.0, 0.05, 0.2, dividend_yield=0.0)
    call_val = bs.call_option_price()
    assert call_val > 0.0


# ---- Tests for US_YieldCurve.py ----

@patch("requests.get")
def test_fetch_us_yield_curve_no_data(mock_get):
    """
    If no yield data is present in the XML, function should return empty arrays.
    """
    xml_content = b"<root><UnknownTag></UnknownTag></root>"
    mock_response = MagicMock()
    mock_response.content = xml_content
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    mats, yields, dt = fetch_us_yield_curve_with_maturities()
    assert mats == []
    assert yields == []
    assert dt is None


@patch("requests.get")
def test_fetch_us_yield_curve_http_error(mock_get):
    """
    If there's an HTTP error, the function should return empty arrays.
    """
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.RequestException("HTTP Error")
    mock_get.return_value = mock_response
    mats, yields, dt = fetch_us_yield_curve_with_maturities()
    assert mats == []
    assert yields == []
    assert dt is None


# ---- Tests for calculate_risk_free_rate_from_yield_curve in streamlit_app.py ----

def test_calculate_risk_free_rate_no_data():
    """
    If no yield curve data is provided, fallback = 0.05.
    """
    result = calculate_risk_free_rate_from_yield_curve(([], []), 1.0)
    assert isclose(result, 0.05, rel_tol=1e-8)


def test_calculate_risk_free_rate_below_first():
    """
    If time_to_maturity < maturities[0], we return yields[0].
    """
    mat = [2, 3, 5]
    yld = [0.02, 0.03, 0.04]
    # T=1 < 2
    result = calculate_risk_free_rate_from_yield_curve((mat, yld), 1.0)
    assert isclose(result, 0.02, rel_tol=1e-8)


def test_calculate_risk_free_rate_above_last():
    """
    If time_to_maturity > maturities[-1], we return yields[-1].
    """
    mat = [1, 2, 3]
    yld = [0.01, 0.02, 0.03]
    # T=4 > 3
    result = calculate_risk_free_rate_from_yield_curve((mat, yld), 4.0)
    assert isclose(result, 0.03, rel_tol=1e-8)


def test_calculate_risk_free_rate_interpolation():
    """
    Test linear interpolation between two points.
    Suppose we have T1=1, T2=2 => yields=0.01, 0.03
    For T=1.5, we expect yield=0.02
    """
    mat = [1, 2]
    yld = [0.01, 0.03]
    result = calculate_risk_free_rate_from_yield_curve((mat, yld), 1.5)
    assert isclose(result, 0.02, rel_tol=1e-8)