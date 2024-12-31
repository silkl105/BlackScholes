import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


class BlackScholes:
    """
    A class to compute Black-Scholes option prices and generate heatmaps of
    call and put prices across a range of spot prices and volatilities.

    All numerical parameters must be strictly greater than zero.

    Parameters
    ----------
    current_price : float, The current price of the underlying asset, must be > 0.
    strike : float, The strike price of the option, must be > 0.
    time_to_maturity : float, Time to maturity (in years), must be > 0.
    interest_rate : float, The risk-free interest rate (annualized), must be > 0.
    volatility : float, The volatility (annualized) of the underlying asset, must be > 0.
    """

    def __init__(
        self,
        current_price: float,
        strike: float,
        time_to_maturity: float,
        interest_rate: float,
        volatility: float
    ) -> None:
        # Validate that all parameters are > 0
        inputs = {"current_price": current_price, "strike": strike, "time_to_maturity": time_to_maturity, "interest_rate": interest_rate, "volatility": volatility}
        for name, value in inputs.items():
            if value <= 0:
                raise ValueError(
                    f"{name} must be a positive float, got {value} instead."
                )

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
        tuple[float, float]: A tuple containing (d1, d2).
        """
        d1 = (
            np.log(self.current_price / self.strike)
            + (self.interest_rate + 0.5 * self.volatility ** 2)
              * self.time_to_maturity
        ) / (self.volatility * np.sqrt(self.time_to_maturity))

        d2 = d1 - self.volatility * np.sqrt(self.time_to_maturity)
        return d1, d2

    def call_option_price(self) -> float:
        """
        Compute the Black-Scholes call option price.

        Returns
        -------
        float: The call option price.
        """
        d1, d2 = self.calculate_d1_d2()
        discounted_strike = (
            self.strike
            * np.exp(-self.interest_rate * self.time_to_maturity)
        )
        return (self.current_price * norm.cdf(d1)) - (
            discounted_strike * norm.cdf(d2)
        )

    def put_option_price(self) -> float:
        """
        Compute the Black-Scholes put option price.

        Returns
        -------
        float: The put option price.
        """
        d1, d2 = self.calculate_d1_d2()
        discounted_strike = (
            self.strike
            * np.exp(-self.interest_rate * self.time_to_maturity)
        )
        return (discounted_strike * norm.cdf(-d2)) - (
            self.current_price * norm.cdf(-d1)
        )

    def generate_heatmaps(
        self,
        spot_range: tuple[float, float],
        volatility_range: tuple[float, float]
    ) -> None:
        """
        Generate call and put price heatmaps over specified spot-price and
        volatility ranges. Both lower bounds (spot_range[0] and
        volatility_range[0]) must be > 0.

        The grid size is determined as follows:
        - If spot range width >= 5, use 10 columns; otherwise subdivide in steps of 0.5.
        - If volatility range width >= 0.10, use 10 rows; otherwise subdivide in steps of 0.01.
        - We round the differences to 4 decimals to avoid floating-point precision issues.

        Parameters
        ----------
        spot_range : tuple[float, float]: The minimum and maximum spot prices to cover (inclusive). Both must be > 0.
        volatility_range : tuple[float, float]: The minimum and maximum volatilities to cover (inclusive). Both must be > 0.

        Raises
        ------
        ValueError
            If the lower bound of either range is not strictly > 0.
        """
        # Validate that bottom values of the ranges are > 0
        if spot_range[0] <= 0:
            raise ValueError(
                f"spot_range lower bound must be > 0, got {spot_range[0]} instead."
            )
        if volatility_range[0] <= 0:
            raise ValueError(
                f"volatility_range lower bound must be > 0, got {volatility_range[0]} instead."
            )

        # Determine how many steps for spot and volatility
        ds = round(spot_range[1] - spot_range[0], 4)
        dv = round(volatility_range[1] - volatility_range[0], 4)

        # Spot resolution
        if ds >= 5.0:
            x_res = 10
        else:
            intervals_spot = int(np.floor(ds / 0.5))
            x_res = max(2, intervals_spot + 1)

        # Volatility resolution
        if dv >= 0.10:
            y_res = 10
        else:
            intervals_vol = int(np.floor(dv / 0.01))
            y_res = max(2, intervals_vol + 1)

        # Construct the arrays of spot prices and volatilities
        spot_prices = np.linspace(spot_range[0], spot_range[1], x_res)
        volatilities = np.linspace(volatility_range[0], volatility_range[1], y_res)

        # Compute call/put prices in a 2D grid
        call_prices = np.zeros((y_res, x_res))
        put_prices = np.zeros((y_res, x_res))

        for i, vol in enumerate(volatilities):
            self.volatility = vol
            for j, sp in enumerate(spot_prices):
                self.current_price = sp
                call_prices[i, j] = self.call_option_price()
                put_prices[i, j] = self.put_option_price()

        # Plotting setup
        fig, (ax_call, ax_put) = plt.subplots(1, 2, figsize=(14, 6))
        plt.subplots_adjust(wspace=0.25)

        # Call Heatmap
        sns.heatmap(
            call_prices,
            xticklabels=[f"{sp:.1f}" for sp in spot_prices],
            yticklabels=[f"{v:.2f}" for v in volatilities],
            annot=True,
            annot_kws={"size": 8},
            fmt=".2f",
            cmap="viridis",
            ax=ax_call
        )
        ax_call.set_title("Call Price Heatmap")
        ax_call.set_xlabel("Spot Price")
        ax_call.set_ylabel("Volatility")
        ax_call.set_xticklabels(ax_call.get_xticklabels(), rotation=0)

        # Put Heatmap
        sns.heatmap(
            put_prices,
            xticklabels=[f"{sp:.1f}" for sp in spot_prices],
            yticklabels=[f"{v:.2f}" for v in volatilities],
            annot=True,
            annot_kws={"size": 8},
            fmt=".2f",
            cmap="viridis",
            ax=ax_put
        )
        ax_put.set_title("Put Price Heatmap")
        ax_put.set_xlabel("Spot Price")
        ax_put.set_ylabel("Volatility")
        ax_put.set_xticklabels(ax_put.get_xticklabels(), rotation=0)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 1) Valid inputs, large ranges => 10 x 10
    bs = BlackScholes(130.0, 100.0, 3.0, 0.04, 0.3)
    bs.generate_heatmaps((1.0, 156.0), (0.15, 0.45))

    # 2) Narrower volatility => dynamic steps
    bs.generate_heatmaps((100.0, 103.5), (0.10, 0.12))

    # 3) Another narrower example
    bs.generate_heatmaps((100.0, 110.0), (0.10, 0.14))