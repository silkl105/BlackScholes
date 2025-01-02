import streamlit as st
import pandas as pd
from BlackScholes import BlackScholes
import warnings
st.set_page_config(layout="wide")

def main():
    """
    The main Streamlit application function for Black-Scholes option pricing.
    Presents sidebar inputs for model parameters, displays the resulting call/put
    prices, and shows separate heatmaps for each with optional purchase price
    toggles (for PnL calculations).
    """

    # --- SIDEBAR LAYOUT ---
    st.sidebar.markdown("## **Black Scholes Option Pricing Project**")
    st.sidebar.markdown("<span style='color: green;'>Created by:</span>", unsafe_allow_html=True)
    st.sidebar.markdown(
        '<a href="https://www.linkedin.com/in/silvioklein/" target="_blank" style="text-decoration: none; color: inherit;">'
        '<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" '
        'style="vertical-align: middle; margin-right: 8px;">Silvio Klein</a>', 
        unsafe_allow_html=True
    )

    st.sidebar.markdown("### Option Price Inputs")
    current_price = st.sidebar.number_input("Current Asset Price", min_value=0.01, value=100.0)
    strike_price = st.sidebar.number_input("Strike Price", min_value=0.01, value=100.0)
    time_to_maturity = st.sidebar.number_input("Time to Maturity (years)", min_value=0.01, value=1.0)
    interest_rate = st.sidebar.number_input("Risk-Free Interest Rate", min_value=0.0, max_value=1.0, value=0.05)
    volatility = st.sidebar.number_input("Volatility", min_value=0.01, value=0.2)

    # Heatmap Parameter Sliders
    st.sidebar.markdown("---")
    st.sidebar.subheader("Heatmap Parameters")
    vol_min = st.sidebar.slider("Min Volatility", 0.01, 1.0, 0.1, 0.01)
    vol_max = st.sidebar.slider("Max Volatility", 0.01, 1.0, 0.5, 0.01)
    spot_min = st.sidebar.number_input("Min Spot Price", min_value=0.01, value=current_price * 0.8)
    spot_max = st.sidebar.number_input("Max Spot Price", min_value=0.01, value=current_price * 1.2)

    # --- MAIN PAGE LAYOUT ---
    st.title("Black-Scholes Pricing Model")

    # Instantiate the BlackScholes model
    bs = BlackScholes(
        current_price=current_price,
        strike=strike_price,
        time_to_maturity=time_to_maturity,
        interest_rate=interest_rate,
        volatility=volatility
    )

    # Calculate current call and put prices
    call_price = bs.call_option_price()
    put_price = bs.put_option_price()

    # Display the user inputs in a table
    input_data = {
        "Current Asset Price": [current_price],
        "Strike Price": [strike_price],
        "Time to Maturity (Years)": [time_to_maturity],
        "Volatility (σ)": [volatility],
        "Risk-Free Interest Rate": [interest_rate],
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

    c1, c2 = st.columns([1, 1], gap="small")
    with c1:
        st.markdown(f"""
            <div class="metric-container metric-call">
                <div>
                    <div class="metric-label">CALL Value</div>
                    <div class="metric-value">${call_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="metric-container metric-put">
                <div>
                    <div class="metric-label">PUT Value</div>
                    <div class="metric-value">${put_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

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
            help="If > 0, heatmap shows (call value - purchase price call). Otherwise, it shows the raw call price."
        )
        # Only affect the call heatmap
        fig_call, _ = bs.generate_heatmaps(
            (spot_min, spot_max),
            (vol_min, vol_max),
            purchase_price_call=purchase_price_call,
            purchase_price_put=0.0
        )
        st.pyplot(fig_call)

    with heat_col2:
        purchase_price_put = st.number_input(
            "Put Purchase Price",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            value=0.0,
            help="If > 0, heatmap shows (put value - purchase price put). Otherwise, it shows the raw put price."
        )
        # Only affect the put heatmap
        _, fig_put = bs.generate_heatmaps(
            (spot_min, spot_max),
            (vol_min, vol_max),
            purchase_price_call=0.0,
            purchase_price_put=purchase_price_put
        )
        st.pyplot(fig_put)

if __name__ == "__main__":
    main()