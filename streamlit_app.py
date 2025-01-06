import streamlit as st
import pandas as pd
from BlackScholes import BlackScholes
import plotly.express as px
import warnings
import os
st.set_page_config(layout="wide")

def main():
    """
    The main Streamlit application function for Black-Scholes option pricing.
    Presents sidebar inputs for model parameters, displays the resulting call/put
    prices, and shows separate heatmaps for each with optional purchase price
    toggles (for PnL calculations).
    """

    # --- SIDEBAR LAYOUT ---
    st.sidebar.markdown("## **Black-Scholes-Merton Option Pricing Project**")
    st.sidebar.markdown("<span style='color: green;'>Created by:</span>", unsafe_allow_html=True)
    st.sidebar.markdown(
        '<a href="https://www.linkedin.com/in/silvioklein/" target="_blank" style="text-decoration: none; color: inherit;">'
        '<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" '
        'style="vertical-align: middle; margin-right: 8px;">Silvio Klein</a>', 
        unsafe_allow_html=True
    )

    st.sidebar.markdown("### Option Price Inputs")
    underlying_price = st.sidebar.number_input("Current Asset Price (S)", min_value=0.01, value=100.0, help="The current (spot) price of the underlying asset ($ per share), must be > 0.")
    strike_price = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0, help = "The strike price of the option ($ per share), must be > 0.")
    time_to_maturity = st.sidebar.number_input("Time to Maturity (years, t)", min_value=0.01, value=1.0, help ="The time to maturity in years, must be > 0.")
    interest_rate = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.0, max_value=1.0, value=0.05, help = "The continuously compounded risk-free interest rate (% p.a.), must be > 0.")
    volatility = st.sidebar.number_input("Volatility (σ)", min_value=0.01, value=0.2, help ="The volatility of the underlying asset (% p.a.), must be > 0.")
    dividend_yield = st.sidebar.number_input("Dividend Yield (q)", min_value=0.0, max_value=1.0, value=0.0, help="Continuously compounded  dividend yield (% p.a., 0.00 implies no dividends).")

    # Heatmap Parameter Sliders
    st.sidebar.markdown("---")
    st.sidebar.subheader("Heatmap Parameters")
    vol_min = st.sidebar.slider("Min Volatility", 0.01, 1.0, 0.1, 0.01)
    vol_max = st.sidebar.slider("Max Volatility", 0.01, 1.0, 0.5, 0.01)
    spot_min = st.sidebar.number_input("Min Spot Price", min_value=0.01, value=underlying_price * 0.8)
    spot_max = st.sidebar.number_input("Max Spot Price", min_value=0.01, value=underlying_price * 1.2)

    # --- MAIN PAGE LAYOUT ---
    st.title("Black-Scholes-Merton Pricing Model")

    # Instantiate the BlackScholes model
    bs = BlackScholes(
        underlying_price=underlying_price,
        strike=strike_price,
        time_to_maturity=time_to_maturity,
        interest_rate=interest_rate,
        volatility=volatility,
        dividend_yield=dividend_yield
    )

    # Calculate current call and put prices, and greeks
    call_price = bs.call_option_price()
    put_price = bs.put_option_price()
    greeks_dict = bs.greeks()

    # Display the user inputs in a table
    input_data = {
        "Current Asset Price": [underlying_price],
        "Strike Price": [strike_price],
        "Time to Maturity (Years)": [time_to_maturity],
        "Volatility (σ)": [volatility],
        "Risk-Free Interest Rate": [interest_rate],
        "Dividend Yield (q)": [dividend_yield],  # new
    }
    st.table(pd.DataFrame(input_data).reset_index(drop=True))

    # Display CALL & PUT values in color-coded metric containers
    st.markdown("""
    <style>
    .metric-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 8px;
        width: auto;
        margin: 0 auto;
    }
    .metric-greeks {
        border: 1px solid white; /* white border */
        padding: 8px;
        margin-top: 10px; 
        border-radius: 5px; 
    }
    .metric-call {
        background-color: #90ee90;
        color: black;
        margin-right: 10px;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
    }
    .metric-put {
        background-color: #ffcccb;
        color: black;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 900;
        margin: 0;
    }
    .metric-label {
        font-size: 1.2rem;
        font-weight: normal;
        margin-bottom: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Greeks
    call_delta = greeks_dict["delta_call"]
    call_theta = greeks_dict["theta_call"]
    put_delta  = greeks_dict["delta_put"]
    put_theta  = greeks_dict["theta_put"]
    gamma      = greeks_dict["gamma"]
    vega       = greeks_dict["vega"]

    c1, c2 = st.columns([1, 1], gap="small")

    with c1:
        # "Call Value" box
        st.markdown(f"""
            <div class="metric-container metric-call">
                <div>
                    <div class="metric-label">CALL Value</div>
                    <div class="metric-value">${call_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # White-bordered box for Call Greeks:
        st.markdown(f"""
            <div class="metric-greeks">
                <p style="margin: 0; color: white;">Delta: {call_delta:.4f}</p>
                <p style="margin: 0; color: white;">Gamma: {gamma:.4f}</p>
                <p style="margin: 0; color: white;">Theta: {call_theta:.4f}</p>
                <p style="margin: 0; color: white;">Vega: {vega:.4f}</p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        # "Put Value" box
        st.markdown(f"""
            <div class="metric-container metric-put">
                <div>
                    <div class="metric-label">PUT Value</div>
                    <div class="metric-value">${put_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # White-bordered box for Put Greeks:
        st.markdown(f"""
            <div class="metric-greeks">
                <p style="margin: 0; color: white;">Delta: {put_delta:.4f}</p>
                <p style="margin: 0; color: white;">Gamma: {gamma:.4f}</p>
                <p style="margin: 0; color: white;">Theta: {put_theta:.4f}</p>
                <p style="margin: 0; color: white;">Vega: {vega:.4f}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("More about Greeks:"):
        st.info("""
        - **Delta (Δ):** The first derivative of option price with respect to underlying price S. Measures the sensitivity to underlying price changes, i.e., how much the option price changes when the underlying price changes by $1.
        - **Gamma (Γ):** The second derivative of option price with respect to underlying price S. Measures the rate of change of Delta itself, i.e., how much delta changes when the underlying price changes by $1.
        - **Theta (Θ):** The first derivative of option price with respect to time to expiration t. Measures the option price time sensitivity/time decay, i.e., how much the option price decreases as 1 calendar day passes.
        - **Vega:** The first derivative of option price with respect to volatility σ. Measures how sensitive the option price is to a 1 percentage point change in volatility.
        """)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Explore How Greeks Respond to Changing Option Parameters:")

        # 1) Two dropdowns: which Greek, which parameter
        greek_options = [("Call Delta", "delta_call"),
                        ("Put Delta", "delta_put"),
                        ("Gamma", "gamma"),
                        ("Call Theta", "theta_call"),
                        ("Put Theta", "theta_put"),
                        ("Vega", "vega")]
        param_options = [("Spot Price (S)", "underlying_price"),
                        ("Volatility (σ)", "volatility"),
                        ("Time to Maturity (t)", "time_to_maturity"),
                        ("Interest Rate (r)", "interest_rate")]

        selected_greek_label = st.selectbox(
            "Select Greek (y-axis)",
            [opt[0] for opt in greek_options]
        )
        selected_param_label = st.selectbox(
            "Select parameter to vary (x-axis)",
            [opt[0] for opt in param_options]
        )
        
        # Map the user-friendly label to the internal key
        greek_internal = next(code for label, code in greek_options if label == selected_greek_label)
        param_internal = next(code for label, code in param_options if label == selected_param_label)

        # 2) Call bs.compute_greek_curve(...) with the chosen param & Greek
        x_vals, y_vals = bs.compute_greek_curve(greek_name=greek_internal, param_name=param_internal)

        # 3) Plot with plotly or streamlit line chart
        fig = px.line(
            x=x_vals, y=y_vals,
            labels={"x": selected_param_label, "y": selected_greek_label},
            title=f"{selected_greek_label} vs. {selected_param_label}"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("Call and Put Price Heatmaps")
    st.info(
        "Explore how European option prices differ based on spot prices and volatility.\n\n"
        "If you enter a purchase price > 0, the heatmap shows PnL values (option price - purchase price) "
        "using a diverging red-green colormap centered at zero. Otherwise, it displays raw option prices."
    )

    # Two columns: each with its own purchase price and heatmap
    heat_col1, heat_col2 = st.columns(2)

    with heat_col1:
        purchase_price_call = st.number_input(
            "Call Purchase Price",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            value=0.0,
            help="If > 0.00, the heatmap shows the call price - the purchase price. Otherwise, it shows the raw call price."
        )
        # Only affect the call heatmap
        fig_call, _ = bs.generate_heatmaps(
            (spot_min, spot_max),
            (vol_min, vol_max),
            purchase_price_call=purchase_price_call,
            purchase_price_put=0.0
        )
        st.plotly_chart(fig_call, use_container_width=True)

    with heat_col2:
        purchase_price_put = st.number_input(
            "Put Purchase Price",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            value=0.0,
            help="If > 0.00, the heatmap shows the put price - the purchase price. Otherwise, it shows the raw put price."
        )
        # Only affect the put heatmap
        _, fig_put = bs.generate_heatmaps(
            (spot_min, spot_max),
            (vol_min, vol_max),
            purchase_price_call=0.0,
            purchase_price_put=purchase_price_put
        )
        st.plotly_chart(fig_put, use_container_width=True)

if __name__ == "__main__":
    main()