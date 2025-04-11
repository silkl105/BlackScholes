# Black-Scholes-Merton Option Pricing and US Yield Curve Integration

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11.8-blue?style=for-the-badge&logo=python" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/Streamlit-1.41.1-ff69b4?style=for-the-badge&logo=streamlit" alt="Streamlit 1.41" />
  <img src="https://img.shields.io/badge/Anaconda-Enabled-44a833?style=for-the-badge&logo=anaconda" alt="Anaconda Compatible" />
  <a href="https://codecov.io/gh/silkl105/BlackScholes">
    <img src="https://codecov.io/gh/silkl105/BlackScholes/graph/badge.svg?token=C0U8LI5RJ0" alt="codecov" style="vertical-align:middle;"/>
  </a>
</p>

A Python-based project that implements the **Black-Scholes-Merton** model for pricing European call and put options, along with functionality to visualize option Greeks, generate heatmaps, and optionally incorporate real-world **US Treasury yield curve** data. 

<table>
  <tr>
    <td><img src="https://cdn-icons-png.flaticon.com/512/3063/3063824.png" width="50" /></td>
    <td>A fully interactive <strong>Streamlit</strong> web application is included so you can explore how option prices and Greeks are affected by spot prices, volatility, dividends, and interest rates. You can try the app live on Streamlit Cloud: [application-blackscholes.streamlit.app](https://application-blackscholes.streamlit.app)</td>
  </tr>
</table>

---

## 📊 Features

- **Black-Scholes-Merton Model:** Calculates option prices (Call & Put) and Greeks (Delta, Gamma, Vega, Theta) for European-style options.

- **Dividend Yield Support:** Includes the continuously compounded dividend yield in calculations (B-S-M extended model).

- **US Treasury Yield Curve Integration:**  
  - Fetches the latest XML data from the U.S. Department of the Treasury.
  - Converts yields from semi-annual compounding to continuously compounded form.
  - Interpolates maturities from 1 month out to 30 years.
  - Allows for either a manually specified risk-free rate or an automatically interpolated rate from the yield curve.

- **Visualizations:**  
  - **Heatmaps** of option prices or PnL across a user-defined range of spot prices and volatilities.
  - **Line Plots** of Greeks vs. model parameters (spot, volatility, time to maturity, interest rate).
  - An interactive chart for the **US yield curve**.

- **Testing with Pytest:** Comprehensive tests (`test_scripts.py`) ensure model correctness and code reliability.

---

## ⚙️ Installation

1. **Clone** this repository or download the files.
2. Ensure you have [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
3. **Navigate** to the project folder where `environment.yaml` is located.
4. **Create and activate** the conda environment. The environment name is set to envs in environment.yaml. Feel free to rename it inside the YAML file or after creation if needed.

   ```bash
   conda env create -f environment.yaml
   conda activate envs

---

## 🚀 Usage

### Running the Streamlit App

1. Ensure the conda environment is activated (`conda activate envs`).
2. From the project root, run:

   ```bash
   streamlit run streamlit_app.py
3. Open your web browser and navigate to the URL printed in your terminal.


**Key Steps in the App:**

- **Left Sidebar:** Provide the Spot Price (S), Strike Price (K), Volatility (σ), Time to Maturity (T), Risk-Free Rate (r), and Dividend Yield (q).
- **Use US Yield Curve:** Optionally enable the “Use US Yield Curve” checkbox to interpolate the rate from the most recent yield curve data.
- **Tabs:**  
  - **Option Prices & Greeks:** Explore various Greeks and their sensitivity to model parameters.
  - **Call and Put Price Heatmaps:** Visualize heatmaps for call and put options under user-defined ranges of spot and volatility.
  -	**US Yield Curve Visualization:** Check out the US Treasury yield curve tab to see how the yield data was retrieved and converted.

### Running Tests
To validate the library:

  ```bash
  pytest test_scripts.py
```

All tests are located in the test_scripts.py file. This ensures correct implementation of option pricing, Greeks, yield curve retrieval, and risk-free rate interpolation.
The tests cover the following:
- Price and Greek calculations
- Input validation
- Yield curve fetching & fallback logic
- Sensitivity curve generation
- Exception handling

---

## 📁 Project Structure

        BlackScholes/
        ├── environment.yaml        <-- Conda environment definition
        ├── BlackScholes.py         <-- Main Black-Scholes-Merton class
        ├── US_YieldCurve.py        <-- US Treasury yield curve fetching & interpolation
        ├── streamlit_app.py        <-- Streamlit web application
        ├── test_scripts.py         <-- Pytest test suite
        └── README.md               <-- Project documentation

- **BlackScholes.py:** Implements closed-form solutions for European call and put options, including Greeks like Delta, Gamma, Vega, and Theta. It also supports:
  - Continuous dividend yield
  - Yield curve-based interest rate lookup
  - Heatmap generation
  - Sensitivity (Greek) curves

- **US_YieldCurve.py:** Fetches and interpolates the [Daily U.S. Treasury Par Yield Curve](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2025) from the official XML feed, converting from CMT yields to continuously-compounded yields.

- **streamlit_app.py:** Streamlit front-end where users can interact with the model parameters and see results rendered in real time.

- **test_scripts.py:** A comprehensive test suite using pytest to validate pricing logic, Greeks, edge cases, yield data handling, and Streamlit helper functions.

---

## 💡 Future Enhancements

Here are some ideas for future development:

1. Advanced Models: Add support for the Heston or SABR models to capture stochastic volatility.
2. American Options Extension: Integrate a numerical method (like binomial trees) to value American-style options.
3. Calibration to Market Data: Implement a calibration feature allowing users to calibrate volatility from market option prices.
4. Historical Volatility: Automated retrieval of historical price data to estimate volatility.
5. Enhanced Data Sources: Expand to additional yield curves (e.g., EUR, GBP).

---

### ⚠️ Disclaimer
This tool is intended for illustrative purposes only. It is not intended for trading, financial advice, or use in production environments. Use at your own risk.

<br>
<p align="center">
  <b>Built by:</b><br>
  <a href="https://www.linkedin.com/in/silvioklein/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-Silvio%20Klein-blue?style=flat&logo=linkedin" />
  </a>
</p>
