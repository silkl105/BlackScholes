import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import plotly.graph_objects as go


class BlackScholes:
    """
    A class to compute Black-Scholes option prices and generate heatmaps of
    call and put prices (or PnL if a purchase price is given) across a range
    of spot prices and volatilities.

    If a user-specified 'purchase_price_call' or 'purchase_price_put' is > 0,
    the heatmap will display the PnL, defined by: PnL = option_value - purchase_price
    using a diverging colormap (green for positive, red for negative). 
    Otherwise, if purchase prices are 0.0, the heatmaps show the standard 
    option values with the 'viridis' colormap.

    Parameters
    ----------
    underlying_price: float. The current (spot) price of the underlying asset ($ per share), must be > 0.
    strike: float. The strike price of the option ($ per share), must be > 0.
    time_to_maturity: float. The time to maturity in years, must be > 0.
    interest_rate: float. The continuously compounded risk-free interest rate (% p.a.), must be > 0.
    volatility: float. The volatility of the underlying asset (% p.a.), must be > 0.
    dividend_yield : float, optional. If > 0, the Black-Scholes-Merton model includes the continuously compounded dividend yield parameter (% p.a.). Default is 0.00.

    Raises
    ------
    ValueError. If dividend_yield is negative or if any of the other initialized parameters is non-positive.
    """

    def __init__(self, underlying_price: float, strike: float, time_to_maturity: float, interest_rate: float, volatility: float, dividend_yield: float = 0.0) -> None:
        # Validate that all parameters are > 0
        inputs = {"underlying_price": underlying_price,
                  "strike": strike,
                  "time_to_maturity": time_to_maturity,
                  "interest_rate": interest_rate,
                  "volatility": volatility
        }
        for name, value in inputs.items():
            if value <= 0:
                raise ValueError(f"{name} must be a positive float, got {value} instead.")
        
        if dividend_yield < 0:
            raise ValueError(f"dividend_yield must be >= 0, got {dividend_yield} instead.")

        self.underlying_price = underlying_price
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.interest_rate = interest_rate
        self.volatility = volatility
        self.dividend_yield = dividend_yield

    def calculate_d1_d2(self) -> tuple[float, float]:
        """
        Calculate the d1 and d2 parameters used in the Black-Scholes-Merton formula.

        Returns
        -------
        (d1, d2): tuple of floats. The d1 and d2 values for the Black-Scholes-Merton formula.
        """
        d1 = (np.log(self.underlying_price / self.strike) + ((self.interest_rate - self.dividend_yield) + 0.5 * self.volatility**2)
              * self.time_to_maturity) / (self.volatility * np.sqrt(self.time_to_maturity))

        d2 = d1 - self.volatility * np.sqrt(self.time_to_maturity)
        return d1, d2

    def call_option_price(self) -> float:
        """
        Compute the Black-Scholes-Merton call option price.

        Returns
        -------
        float. The call option price.
        """
        d1, d2 = self.calculate_d1_d2()
        discounted_strike = self.strike * np.exp(-self.interest_rate * self.time_to_maturity)
        return (
            self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(d1)
            - discounted_strike * norm.cdf(d2)
        )
    
    def put_option_price(self) -> float:
        """
        Compute the Black-Scholes-Merton put option price.

        Returns
        -------
        float. The put option price.
        """
        d1, d2 = self.calculate_d1_d2()
        discounted_strike = self.strike * np.exp(-self.interest_rate * self.time_to_maturity)
        return (
            discounted_strike * norm.cdf(-d2)
            - self.underlying_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(-d1)
        )
    
    def greeks(self) -> dict:
        """
        Compute delta, gamma, theta, and vega for the Call and Put option under the Black-Scholes-Merton model. 

        Returns
        -------
        dictionary. Containing: 'delta_call', 'gamma_call', 'theta_call', 'vega_call',
            'delta_put', 'gamma_put', 'theta_put', 'vega_put'.
        """
        d1, d2 = self.calculate_d1_d2()
        T = self.time_to_maturity
        r = self.interest_rate
        q = self.dividend_yield
        sigma = self.volatility
        S = self.underlying_price
        K = self.strike

        # Probability density at d1
        pdf_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)

        # Delta for call & put: change in option price given a $1 change in spot price
        delta_call = np.exp(-q * T) * norm.cdf(d1)
        delta_put = np.exp(-q * T) * (norm.cdf(d1) - 1.0)

        # Gamma (same for call & put): change in delta given a $1 change in spot price
        gamma = (np.exp(-q * T) * pdf_d1) / (S * sigma * np.sqrt(T))

        # Vega (same for call & put): change in option price given a 1% change in implied volatility
        vega = (1/100) * (S * np.exp(-q * T) * pdf_d1 * np.sqrt(T))

        # Theta for call & put: option time decay, change in option price per one calendar day
        call_theta = (1/365) * (
            - (S * np.exp(-q * T) * pdf_d1 * sigma)
              / (2.0 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )

        put_theta = (1/365) * (
            - (S * np.exp(-q * T) * pdf_d1 * sigma)
              / (2.0 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )

        # Return greeks in a dictionary
        return {
            "delta_call": delta_call,
            "delta_put": delta_put,
            "gamma": gamma,
            "theta_call": call_theta,
            "theta_put": put_theta,
            "vega": vega,
        }
    
    def generate_heatmaps(
        self,
        spot_range: tuple[float, float],
        volatility_range: tuple[float, float],
        purchase_price_call: float = 0.0,
        purchase_price_put: float = 0.0
    ):
        """
        Generate heatmaps for call and put across specified spot and volatility ranges.
        Optionally incorporate a purchase price for each option to display PnL.

        If `purchase_price_call` > 0, the call heatmap will show: call_price - purchase_price_call
        If `purchase_price_put` > 0, the put heatmap will show: put_price - purchase_price_put
        Both are displayed using a diverging colormap (RdYlGn) centered at zero.
        If either purchase price is 0.0, that options heatmap displays its raw price
        with the 'viridis' colormap.

        Parameters
        ----------
        spot_range : tuple of floats.
            (min_spot, max_spot). The inclusive range of spot prices to evaluate. Both must be strictly > 0.
        volatility_range : tuple of floats
            (min_vol, max_vol). The inclusive range of volatilities to evaluate. Both must be strictly > 0.
        purchase_price_call : float, optional. If > 0, display the call heatmap as PnL. Default is 0.0.
        purchase_price_put : float, optional. If > 0, display the put heatmap as PnL. Default is 0.0.

        Returns
        -------
        fig_call : matplotlib.figure.Figure. The Matplotlib figure object for the call heatmap.
        fig_put : matplotlib.figure.Figure. The Matplotlib figure object for the put heatmap.

        Raises
        ------
        ValueError. If the lower bound of either range is not strictly > 0.
        """
        if spot_range[0] <= 0:
            raise ValueError(
                f"spot_range lower bound must be > 0, got {spot_range[0]} instead."
            )
        if volatility_range[0] <= 0:
            raise ValueError(
                f"volatility_range lower bound must be > 0, got {volatility_range[0]} instead."
            )

        # Determine resolution for the spot/x-axis
        ds = round(spot_range[1] - spot_range[0], 4)
        if ds >= 5.0:
            x_res = 10
        else:
            intervals_spot = int(np.floor(ds / 0.5))
            x_res = max(2, intervals_spot + 1)

        # Determine resolution for the volatility/y-axis
        dv = round(volatility_range[1] - volatility_range[0], 4)
        if dv >= 0.10:
            y_res = 10
        else:
            intervals_vol = int(np.floor(dv / 0.01))
            y_res = max(2, intervals_vol + 1)

        # Create arrays of spot prices and volatilities
        spot_prices = np.linspace(spot_range[0], spot_range[1], x_res)
        volatilities = np.linspace(volatility_range[0], volatility_range[1], y_res)

        call_prices = np.zeros((y_res, x_res))
        put_prices = np.zeros((y_res, x_res))

        # Compute option prices across the grid
        for i, vol in enumerate(volatilities):
            self.volatility = vol
            for j, sp in enumerate(spot_prices):
                self.underlying_price = sp
                call_prices[i, j] = self.call_option_price()
                put_prices[i, j] = self.put_option_price()

        # Decide how to render call and put data: PnL or raw option value
        def data_and_cmap(prices: np.ndarray, purchase_price: float):
            """
            Returns (transformed_data, colormap, center).
            If purchase_price > 0, data = prices - purchase_price (PnL), colormap = 'RdYlGn', center=0.
            Otherwise, data = prices (raw option value), colormap = 'viridis', center=None.
            """
            if purchase_price > 0:
                return prices - purchase_price, "RdYlGn", 0
            else:
                return prices, "viridis", None

        call_data, cmap_call, center_call = data_and_cmap(call_prices, purchase_price_call)
        put_data,  cmap_put,  center_put  = data_and_cmap(put_prices, purchase_price_put)

        # --- Create the CALL figure ---
        fig_call, ax_call = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            call_data,
            xticklabels=[f"{p:.1f}" for p in spot_prices],
            yticklabels=[f"{v:.2f}" for v in volatilities],
            annot=True,
            fmt=".2f",
            cmap=cmap_call,
            center=center_call,
            ax=ax_call
        )
        ax_call.set_xlabel("Spot Price")
        ax_call.set_ylabel("Volatility")
        fig_call.tight_layout()

        # --- Create the PUT figure ---
        fig_put, ax_put = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            put_data,
            xticklabels=[f"{p:.1f}" for p in spot_prices],
            yticklabels=[f"{v:.2f}" for v in volatilities],
            annot=True,
            fmt=".2f",
            cmap=cmap_put,
            center=center_put,
            ax=ax_put
        )
        ax_put.set_xlabel("Spot Price")
        ax_put.set_ylabel("Volatility")
        fig_put.tight_layout()

        return fig_call, fig_put
    
    def compute_greek_curve(
        self,
        greek_name: str,
        param_name: str,
        num_points: int = 50,
        pct_range: float = 0.50
    ) -> tuple[list[float], list[float]]:
        """
        Compute how the chosen 'greek_name' changes when 'param_name' is varied from 
        -pct_range% to +pct_range% around its current value. Returns (x_vals, y_vals).

        Dynamically modifies the parameter to generate Greek values while reverting 
        changes after each computation.

        Parameters
        ----------
        greek_name : str
            Must be one of the keys in the dictionary returned by self.greeks(), 
            e.g., "delta_call", "delta_put", "theta_call", "theta_put", "gamma", "vega".
        param_name : str
            Which parameter to vary: "underlying_price", "strike", "time_to_maturity", "interest_rate", or "volatility".
        num_points : int, optional
            Number of sample points in the -pct_range%..+pct_range% interval.
        pct_range : float, optional
            Fractional range (0.50 => ±50%).

        Returns
        -------
        (x_vals, y_vals) : (list of floats, list of floats)
            The parameter values varied on the x-axis and the corresponding Greek values 
            on the y-axis.

        Raises
        ------
        ValueError
            If the provided `param_name` or `greek_name` is invalid.
        """
        # Check if param_name exists
        if not hasattr(self, param_name):
            raise ValueError(f"Invalid parameter name: {param_name}")

        # Get the current value of the parameter to vary
        current_val = getattr(self, param_name)

        # Build an array of parameter values from -pct_range% to +pct_range%
        min_val = current_val * (1 - pct_range)
        max_val = current_val * (1 + pct_range)
        x_vals = np.linspace(min_val, max_val, num_points).tolist()

        # List to store Greek values
        y_vals = []

        # Iterate over the x_vals, temporarily modify the parameter, and compute the Greek
        for x in x_vals:
            setattr(self, param_name, x) # Temporarily override the parameter
            greeks_map = self.greeks()

            if greek_name not in greeks_map:
                raise ValueError(f"Invalid Greek name: {greek_name}")

            y_vals.append(greeks_map[greek_name])
            setattr(self, param_name, current_val) # Revert the parameter back to its original value

        return x_vals, y_vals