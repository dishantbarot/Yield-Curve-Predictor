import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from fredapi import Fred
from datetime import datetime

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.markdown(
    """
    <h1 style='text-align: center;'>📈 Yield Curve Predictor</h1>
    <p style='text-align: center; font-size: 16px; color: gray;'>
        Machine Learning–Driven Analysis of U.S. Treasury Yields
    </p>
    """,
    unsafe_allow_html=True
)

# =====================================================
# API KEY HANDLING (SECURE)
# =====================================================
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

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna()

    return df

df = fetch_yield_data()

# =====================================================
# FEATURE ENGINEERING
# =====================================================
def engineer_features(df):
    df = df.copy()

    # Yield spreads
    df['spread_10y_2y'] = df['10Y'] - df['2Y']
    df['spread_5y_3m'] = df['5Y'] - df['3M']
    df['spread_30y_10y'] = df['30Y'] - df['10Y']

    # Rolling stats
    df['vol_30d'] = df['10Y'].rolling(30).std()
    df['vol_90d'] = df['10Y'].rolling(90).std()
    df['ma_30'] = df['10Y'].rolling(30).mean()
    df['ma_90'] = df['10Y'].rolling(90).mean()

    # Lag features
    for lag in [1, 5, 10]:
        df[f'10Y_lag_{lag}'] = df['10Y'].shift(lag)

    return df.dropna()

df_feat = engineer_features(df)

# =====================================================
# LOAD TRAINED MODEL
# =====================================================
with open("random_forest_package.pkl", "rb") as f:
    model_package = pickle.load(f)

# Extract model and feature list
model = model_package["model"]
feature_cols = model_package["features"]

# =====================================================
# PREDICTION
# =====================================================
latest_features = df_feat[feature_cols].iloc[-1:]
prediction = model.predict(latest_features)[0]

# =====================================================
# LAYOUT (Plot | Insights | Spacer)
# =====================================================
left_col, mid_col, right_col = st.columns([5, 2, 1])


# =====================================================
# YIELD CURVE VISUALIZATION
# =====================================================
with left_col:
    st.subheader("📊 Yield Curve: 2-Year vs 10-Year")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["2Y"],
        name="2Y Yield",
        line=dict(color="#1f77b4", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["10Y"],
        name="10Y Yield",
        line=dict(color="#d62728", width=3)
    ))

    fig.update_layout(
        height=550,                 # 🔥 Increased height
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis_title="Date",
        yaxis_title="Yield (%)",
        hovermode="x unified",
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
# MODEL OUTPUT & SCENARIO ANALYSIS
# =====================================================

with mid_col:
    st.subheader("🔮 Model Insight")

    st.metric(
        label="Predicted Next-Period 10Y Yield",
        value=f"{prediction:.2f} %"
    )

    latest_spread = df["10Y"].iloc[-1] - df["2Y"].iloc[-1]

    st.markdown("### 📌 Curve Signal")
    if latest_spread < 0:
        st.error("Yield Curve Inversion Detected")
        st.caption("Historically associated with recession risk.")
    elif latest_spread < 0.5:
        st.warning("Flattening Yield Curve")
        st.caption("Market uncertainty / late-cycle signal.")
    else:
        st.success("Normal Yield Curve")
        st.caption("Typically associated with economic expansion.")

with right_col:
    st.write("")

# =====================================================
# ECONOMIC EXPLANATION
# =====================================================
st.markdown("""
## 📘 Significance of the Yield Curve

The yield curve represents interest rates across different bond maturities.
It reflects market expectations for economic growth, inflation, and monetary policy.

### 🔴 Yield Curve Inversion
When short-term rates exceed long-term rates (e.g., 2Y > 10Y), it is called an **inversion**.
Historically, this has been one of the most reliable indicators of an upcoming recession.

### 🏠 Impact on Daily Life
- **Mortgage Rates:** Track long-term yields closely
- **Corporate Borrowing:** Higher yields increase financing costs
- **Savings Accounts:** Short-term rates affect deposit returns
- **Inflation:** Reflected in long-term yield expectations
""")

st.caption("⚠️ Educational use only. Not financial advice.")
