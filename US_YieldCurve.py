import requests
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import interp1d
from math import log
from datetime import datetime
from typing import List, Tuple, Optional

YIELD_CURVE_URL = "https://home.treasury.gov/sites/default/files/interest-rates/yield.xml"

def fetch_us_yield_curve_with_maturities(
    url: str = YIELD_CURVE_URL,
    timeout: int = 25
) -> Tuple[List[float], List[float], Optional[datetime]]:
    """
    Fetch the latest US Treasury yield curve data, convert yields to continuously
    compounded rates (see https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/interest-rates-frequently-asked-questions),
    interpolate over continuous maturities, and return the continuous data along
    with the date of the yield curve.

    Parameters
    ----------
    url : str, optional
        The URL of the XML data source for the US Treasury yield curve.
        Defaults to the official treasury.gov link.
    timeout : int, optional. The request timeout in seconds. Defaults to 25.

    Returns
    -------
    tuple of (maturities, yields, latest_date)
        - maturities: list of floats. Continuous maturities in years, roughly from 0.08 to 30 years.
        - yields: list of floats. Interpolated, continuously compounded yields (annualized, in decimal).
        - latest_date: datetime or None. The date of the retrieved yield curve, or None if unavailable.

    Notes
    -----
    - Discrete yields are provided as APRs and converted to continuously compounded rates with: r_cont = ln(1 + apr_decimal).
    - Uses cubic interpolation to get a smooth curve. 
    - Returns empty lists and None if no valid data can be retrieved.
    - If data size becomes large, consider using a streaming parser such as lxml.etree.iterparse for better performance.
    """
    # Mapping for discrete maturities (in years)
    maturity_map = {
        "BC_1MONTH": 1 / 12,
        "BC_2MONTH": 2 / 12,
        "BC_3MONTH": 3 / 12,
        "BC_4MONTH": 4 / 12,
        "BC_6MONTH": 6 / 12,
        "BC_1YEAR": 1,
        "BC_2YEAR": 2,
        "BC_3YEAR": 3,
        "BC_5YEAR": 5,
        "BC_7YEAR": 7,
        "BC_10YEAR": 10,
        "BC_20YEAR": 20,
        "BC_30YEAR": 30,
        "BC_30YEARDISPLAY": 30,
    }

    try:
        # 1. Perform GET request to retrieve the full XML data
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        # 2. Parse the XML
        root = ET.fromstring(response.content)
        data_entries = root.findall(".//G_NEW_DATE")
        if not data_entries:
            # No data found in the XML
            return [], [], None

        latest_date = None
        latest_yields = {}

        # 3. Single pass to find the latest date
        for entry in data_entries:
            date_text = entry.findtext("BID_CURVE_DATE")
            if not date_text:
                continue
            try:
                curve_date = datetime.strptime(date_text, "%d-%b-%y")
            except ValueError:
                continue  # Invalid date format, skip

            bc_cat = entry.find(".//G_BC_CAT")
            if bc_cat is None:
                continue

            # Create a local dict of yields for this date
            local_yields = {}
            for elem in bc_cat:
                if elem.tag in maturity_map and elem.text:
                    try:
                        cmt_rate = float(elem.text) / 100.0  # Convert percentage to decimal
                        apy = (1 + cmt_rate / 2) ** 2 - 1    # Convert semiannual CMT to APY
                        r_cont = log(1 + apy)                # Convert APY to continuously compounded yield
                        local_yields[maturity_map[elem.tag]] = r_cont
                    except ValueError:
                        # If conversion fails, skip this yield
                        continue

            # Update our stored yields if we find a strictly more recent date
            if not latest_date or curve_date > latest_date:
                latest_date = curve_date
                latest_yields = local_yields

        # 4. Interpolate if we have a valid date and yields
        if not latest_yields or not latest_date:
            return [], [], None

        discrete_maturities = sorted(latest_yields.keys())
        discrete_yields = [latest_yields[m] for m in discrete_maturities]

        # Generate a smooth curve from 0.08 to 30 years
        continuous_maturities = np.arange(0.08, 30.01, 0.01)
        interpolation = interp1d(
            discrete_maturities,
            discrete_yields,
            kind="cubic",
            fill_value="extrapolate"
        )
        continuous_yields = interpolation(continuous_maturities)

        return (
            continuous_maturities.tolist(),
            continuous_yields.tolist(),
            latest_date
        )

    except (requests.RequestException, ET.ParseError) as exc:
        # Network error or invalid XML parse
        print(f"[Error] fetch_us_yield_curve_with_maturities: {exc}")
        return [], [], None
    except Exception as exc:
        # Catch-all for any other unexpected errors
        print(f"[Error] Unexpected in fetch_us_yield_curve_with_maturities: {exc}")
        return [], [], None