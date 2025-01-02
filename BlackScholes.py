import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


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
    current_price: float. The current (spot) price of the underlying asset, must be > 0.
    strike: float. The strike price of the option, must be > 0.
    time_to_maturity: float. The time to maturity in years, must be > 0.
    interest_rate: float. The risk-free interest rate (annualized), must be > 0.
    volatility: float. The annualized volatility of the underlying asset, must be > 0.

    Raises
    ------
    ValueError. If any of the initialized parameters is non-positive.
    """

    def __init__(self, current_price: float, strike: float, time_to_maturity: float, interest_rate: float, volatility: float) -> None:
        # Validate that all parameters are > 0
        inputs = {"current_price": current_price,
                  "strike": strike,
                  "time_to_maturity": time_to_maturity,
                  "interest_rate": interest_rate,
                  "volatility": volatility
                  }
        for name, value in inputs.items():
            if value <= 0:
                raise ValueError(f"{name} must be a positive float, got {value} instead.")

        self.current_price = current_price
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.interest_rate = interest_rate
        self.volatility = volatility

    def calculate_d1_d2(self) -> tuple[float, float]:
        """
        Calculate the d1 and d2 parameters used in the Black-Scholes formula.

        Returns
        -------
        (d1, d2): tuple of floats. The d1 and d2 values for the Black-Scholes formula.
        """
        d1 = (np.log(self.current_price / self.strike) + (self.interest_rate + 0.5 * self.volatility ** 2)
              * self.time_to_maturity) / (self.volatility * np.sqrt(self.time_to_maturity))

        d2 = d1 - self.volatility * np.sqrt(self.time_to_maturity)
        return d1, d2

    def call_option_price(self) -> float:
        """
        Compute the Black-Scholes call option price.

        Returns
        -------
        float. The call option price.
        """
        d1, d2 = self.calculate_d1_d2()
        discounted_strike = self.strike * np.exp(-self.interest_rate * self.time_to_maturity)
        return (self.current_price * norm.cdf(d1)) - (discounted_strike * norm.cdf(d2))

    def put_option_price(self) -> float:
        """
        Compute the Black-Scholes put option price.

        Returns
        -------
        float. The put option price.
        """
        d1, d2 = self.calculate_d1_d2()
        discounted_strike = self.strike * np.exp(-self.interest_rate * self.time_to_maturity)
        return (discounted_strike * norm.cdf(-d2)) - (self.current_price * norm.cdf(-d1))

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
                self.current_price = sp
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