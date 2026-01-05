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
st.set_page_config(
    page_title="Yield Curve Analytics",
    page_icon="📈",
    layout="wide"
)

# Professional CSS for Metric Cards and Containers
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 700; color: #1E88E5; }
    .stPlotlyChart { border: 1px solid #30363d; border-radius: 10px; padding: 10px; background-color: #0d1117; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# DATA & ML BACKEND
# =====================================================
try:
    fred = Fred(api_key=st.secrets["FRED_API_KEY"])
except Exception:
    st.error("❌ FRED API key not found. Please configure secrets.toml.")
    st.stop()

@st.cache_data(ttl=3600)
def fetch_yield_data():
    maturities = {
        "1M": "DGS1MO", "3M": "DGS3MO", "1Y": "DGS1", 
        "2Y": "DGS2", "5Y": "DGS5", "10Y": "DGS10", "30Y": "DGS30"
    }
    df = pd.DataFrame({name: fred.get_series(code) for name, code in maturities.items()})
    df.index = pd.to_datetime(df.index)
    return df.sort_index().dropna()

def engineer_features(df):
    """
    RESTORED ALL FEATURES: 
    The model specifically requires these column names to exist.
    """
    df = df.copy()

    # Spreads
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

# Execute Data Pipeline
raw_data = fetch_yield_data()
df_feat = engineer_features(raw_data)

# Load Model
with open("random_forest_package.pkl", "rb") as f:
    model_package = pickle.load(f)

model = model_package["model"]
feature_cols = model_package["features"]

# Prediction using exactly the columns the model expects
latest_features = df_feat[feature_cols].iloc[-1:]
prediction = model.predict(latest_features)[0]

# =====================================================
# HEADER & METRICS
# =====================================================
st.title("📊 Fixed-Income Intelligence Dashboard")
st.markdown("#### **U.S. Treasury Yield Curve ML Forecaster**")
st.divider()

m1, m2, m3, m4 = st.columns(4)
latest_10y = raw_data["10Y"].iloc[-1]
prev_10y = raw_data["10Y"].iloc[-2]
latest_spread = latest_10y - raw_data["2Y"].iloc[-1]

m1.metric("Current 10Y Yield", f"{latest_10y:.2f}%", f"{latest_10y - prev_10y:+.3f}")
m2.metric("ML Predicted (Next)", f"{prediction:.2f}%", f"{prediction - latest_10y:+.2f}")
m3.metric("10Y-2Y Spread", f"{latest_spread:.3f}%")

if latest_spread < 0:
    status, color = "INVERTED", "🔴"
elif latest_spread < 0.5:
    status, color = "FLATTENING", "🟡"
else:
    status, color = "NORMAL", "🟢"
m4.metric("Curve Regime", f"{color} {status}")

# =====================================================
# MAIN PLOT
# =====================================================
st.subheader("Yield Curve Dynamics (2Y vs 10Y)")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=raw_data.index, y=raw_data["10Y"], name="10-Year (Bench)",
    line=dict(color="#00D4FF", width=2.5),
    fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.05)'
))

fig.add_trace(go.Scatter(
    x=raw_data.index, y=raw_data["2Y"], name="2-Year (Short)",
    line=dict(color="#94A3B8", width=1.5, dash='dot')
))

fig.update_layout(
    template="plotly_dark",
    height=550,
    margin=dict(l=0, r=0, t=20, b=0),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=False),
    yaxis=dict(ticksuffix="%", side="right", showgrid=True, gridcolor="#30363d")
)
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PROFESSIONAL SIGNIFICANCE SECTION
# =====================================================
st.divider()
st.subheader("📘 Institutional Knowledge Base")

col_a, col_b = st.columns(2)

with col_a:
    with st.container(border=True):
        st.markdown("### 🏦 Market Significance")
        st.write("""
        The yield curve is the market's primary signaling mechanism for economic health. 
        It plots the interest rates of bonds with equal credit quality but differing maturity dates.
        """)
        st.markdown("**Core Components:**")
        st.info("Short-term rates reflect Fed Policy | Long-term rates reflect Growth/Inflation expectations.")

with col_b:
    with st.container(border=True):
        st.markdown("### 📉 The Inversion Signal")
        st.write("""
        When the spread between 10Y and 2Y yields turns negative, the curve is **Inverted**. 
        Historically, this indicates that investors expect interest rates to fall in the future 
        due to an economic slowdown or recession.
        """)
        

st.caption(f"Last data sync: {datetime.now().strftime('%Y-%m-%d')} | Author: Dishant Barot")
