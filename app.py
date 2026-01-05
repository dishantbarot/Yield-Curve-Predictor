# =====================================================
# YIELD CURVE PREDICTOR — STREAMLIT APPLICATION
# Author: Dishant Barot
# Description:
# A professional FinTech dashboard that fetches U.S. Treasury yields,
# engineers macro-financial features, applies a trained ML model,
# and visualizes yield curve dynamics with economic interpretation.
# =====================================================

# ---------------------------
# IMPORTS
# ---------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from fredapi import Fred
from datetime import datetime


# =====================================================
# PAGE CONFIGURATION & GLOBAL STYLING
# =====================================================
st.set_page_config(
    page_title="Yield Curve Predictor",
    layout="wide"
)

# Global CSS for professional card-style UI
st.markdown(
    """
    <style>
    .metric-card {
        background-color: rgba(255, 255, 255, 0.04);
        padding: 22px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 20px;
    }

    .section-card {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 30px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.10);
        margin-top: 35px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# TITLE & SUBTITLE (CENTER-ALIGNED)
# =====================================================
st.markdown(
    """
    <h1 style="text-align: center;">📈 Yield Curve Predictor</h1>
    <p style="text-align: center; font-size: 16px; color: gray;">
        Machine Learning–Driven Analysis of U.S. Treasury Yields
    </p>
    """,
    unsafe_allow_html=True
)


# =====================================================
# FRED API INITIALIZATION (SECURE)
# =====================================================
# The API key is stored securely in Streamlit secrets
try:
    fred = Fred(api_key=st.secrets["FRED_API_KEY"])
except Exception:
    st.error("❌ FRED API key not found. Please configure secrets.toml.")
    st.stop()


# =====================================================
# DATA ACQUISITION (CACHED)
# =====================================================
@st.cache_data(ttl=3600)
def fetch_yield_data():
    """
    Fetch U.S. Treasury yields from FRED across multiple maturities.
    Data is cached to reduce API calls and improve performance.
    """
    maturities = {
        "1M": "DGS1MO",
        "3M": "DGS3MO",
        "1Y": "DGS1",
        "2Y": "DGS2",
        "5Y": "DGS5",
        "10Y": "DGS10",
        "30Y": "DGS30"
    }

    df = pd.DataFrame()
    for name, code in maturities.items():
        df[name] = fred.get_series(code)

    # Ensure proper time-series format
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna()

    return df


# Load yield data
df = fetch_yield_data()


# =====================================================
# FEATURE ENGINEERING
# =====================================================
def engineer_features(df):
    """
    Create finance-driven features capturing:
    - Yield curve shape
    - Volatility regimes
    - Market memory (lags)
    """
    df = df.copy()

    # Yield spreads (curve shape & recession signals)
    df["spread_10y_2y"] = df["10Y"] - df["2Y"]
    df["spread_5y_3m"] = df["5Y"] - df["3M"]
    df["spread_30y_10y"] = df["30Y"] - df["10Y"]

    # Rolling statistics (trend & volatility)
    df["vol_30d"] = df["10Y"].rolling(30).std()
    df["vol_90d"] = df["10Y"].rolling(90).std()
    df["ma_30"] = df["10Y"].rolling(30).mean()
    df["ma_90"] = df["10Y"].rolling(90).mean()

    # Lag features (market memory)
    for lag in [1, 5, 10]:
        df[f"10Y_lag_{lag}"] = df["10Y"].shift(lag)

    return df.dropna()


# Apply feature engineering
df_feat = engineer_features(df)


# =====================================================
# LOAD TRAINED MACHINE LEARNING MODEL
# =====================================================
# Model is stored as a package:
# {
#   "model": trained_model,
#   "features": feature_column_list
# }
with open("random_forest_package.pkl", "rb") as f:
    model_package = pickle.load(f)

model = model_package["model"]
feature_cols = model_package["features"]


# =====================================================
# MODEL PREDICTION
# =====================================================
# Use the most recent feature row for prediction
latest_features = df_feat[feature_cols].iloc[-1:]
prediction = model.predict(latest_features)[0]


# =====================================================
# MAIN DASHBOARD LAYOUT
# =====================================================
# Plot dominates the left, insights on the right
plot_col, insight_col = st.columns([3.5, 1.5], gap="large")


# =====================================================
# YIELD CURVE VISUALIZATION (HERO ELEMENT)
# =====================================================
with plot_col:
    st.markdown("## Yield Curve Dynamics (2Y vs 10Y Treasury)")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["2Y"],
        name="2-Year Treasury Yield",
        line=dict(color="#1f77b4", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["10Y"],
        name="10-Year Treasury Yield",
        line=dict(color="#d62728", width=3)
    ))

    fig.update_layout(
        height=850,
        margin=dict(l=20, r=20, t=60, b=40),
        xaxis_title="Date",
        yaxis_title="Yield (%)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# MODEL INSIGHTS PANEL (RIGHT SIDE)
# =====================================================
with insight_col:
    # Prediction card
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="color: gray; font-size: 14px;">
                Predicted Next-Period 10Y Yield
            </div>
            <div style="font-size: 36px; font-weight: 600;">
                {prediction:.2f} %
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Yield curve signal interpretation
    latest_spread = df["10Y"].iloc[-1] - df["2Y"].iloc[-1]

    if latest_spread < 0:
        signal = "Inverted Yield Curve"
        interpretation = "Historically associated with elevated recession risk."
    elif latest_spread < 0.5:
        signal = "Flattening Yield Curve"
        interpretation = "Late-cycle signal with increasing economic uncertainty."
    else:
        signal = "Normal Yield Curve"
        interpretation = "Typically observed during periods of economic expansion."

    st.markdown(
        f"""
        <div class="metric-card">
            <div style="color: gray; font-size: 14px;">
                Yield Curve Signal
            </div>
            <div style="font-size: 18px; font-weight: 600; margin-top: 6px;">
                {signal}
            </div>
            <div style="font-size: 13px; color: gray; margin-top: 8px;">
                {interpretation}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# ECONOMIC EXPLANATION SECTION
# =====================================================
st.markdown(
    """
    <div class="section-card">
        <h2>Significance of the Yield Curve</h2>

        <p>
        The yield curve represents interest rates across different bond maturities
        and reflects market expectations for economic growth, inflation,
        and monetary policy.
        </p>

        <h4>Yield Curve Inversion</h4>
        <p>
        When short-term rates exceed long-term rates (e.g., 2Y &gt; 10Y),
        the yield curve becomes inverted — a historically reliable signal
        of upcoming economic slowdowns.
        </p>

        <h4>Impact on Daily Life</h4>
        <ul>
            <li><b>Mortgage Rates:</b> Closely track long-term yields</li>
            <li><b>Corporate Borrowing:</b> Higher yields increase financing costs</li>
            <li><b>Savings Accounts:</b> Short-term rates affect deposit returns</li>
            <li><b>Inflation:</b> Reflected in long-term yield expectations</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True   
)

