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
    page_title="Yield Curve Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a refined look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #1f77b4; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# SIDEBAR & SETTINGS
# =====================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2620/2620582.png", width=80)
    st.title("Control Panel")
    st.markdown("---")
    
    # Secure API Key Entry (If not in secrets)
    api_key = st.text_input("FRED API Key", type="password", value=st.secrets.get("FRED_API_KEY", ""))
    
    st.info("This tool uses Random Forest Regressors to predict future Treasury yields based on historical spreads and volatility.")
    
    st.markdown("---")
    st.caption("v2.1.0 | Data source: FRED")

# Stop if no API key
if not api_key:
    st.error("Please enter a FRED API key in the sidebar or secrets.toml to proceed.")
    st.stop()

fred = Fred(api_key=api_key)

# =====================================================
# DATA & FEATURES (Optimized)
# =====================================================
@st.cache_data(ttl=3600)
def fetch_and_process():
    maturities = {"1M": "DGS1MO", "3M": "DGS3MO", "1Y": "DGS1", "2Y": "DGS2", "5Y": "DGS5", "10Y": "DGS10", "30Y": "DGS30"}
    df = pd.DataFrame({name: fred.get_series(code) for name, code in maturities.items()})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna()
    
    # Feature Engineering
    df['spread_10y_2y'] = df['10Y'] - df['2Y']
    df['vol_30d'] = df['10Y'].rolling(30).std()
    df['ma_30'] = df['10Y'].rolling(30).mean()
    for lag in [1, 5]:
        df[f'10Y_lag_{lag}'] = df['10Y'].shift(lag)
    return df.dropna()

df_feat = fetch_and_process()

# =====================================================
# HEADER SECTION
# =====================================================
row0_1, row0_2 = st.columns([3, 1])
with row0_1:
    st.title("📈 Yield Curve Predictor")
    st.subheader("Machine Learning Analysis of U.S. Treasury Yields")

# =====================================================
# METRIC CARDS
# =====================================================
curr_10y = df_feat["10Y"].iloc[-1]
prev_10y = df_feat["10Y"].iloc[-2]
curr_spread = df_feat["spread_10y_2y"].iloc[-1]

# Prediction logic
with open("random_forest_package.pkl", "rb") as f:
    model_package = pickle.load(f)
prediction = model_package["model"].predict(df_feat[model_package["features"]].iloc[-1:])[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current 10Y Yield", f"{curr_10y:.2f}%", f"{(curr_10y - prev_10y):.3f}")
m2.metric("10Y-2Y Spread", f"{curr_spread:.2f}%")
m3.metric("ML Predicted (Next)", f"{prediction:.2f}%", delta_color="off")

# Curve Status Logic
if curr_spread < 0:
    status, color = "Inverted", "normal"
elif curr_spread < 0.5:
    status, color = "Flattening", "off"
else:
    status, color = "Healthy", "normal"
m4.metric("Curve Status", status)

st.markdown("---")

# =====================================================
# MAIN CONTENT TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(["📊 Market Analysis", "🤖 Model Logic", "📘 Economic Guide"])

with tab1:
    col_chart, col_side = st.columns([3, 1])
    
    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["2Y"], name="2Y Yield", line=dict(color="#94a3b8", width=1.5)))
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["10Y"], name="10Y Yield", line=dict(color="#2563eb", width=2.5)))
        
        fig.update_layout(
            height=500,
            margin=dict(l=0, r=0, t=20, b=0),
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.write("### Quick Insights")
        st.info(f"**Signal:** {status}")
        st.write(f"The current spread is **{curr_spread:.2f}**. Historically, a negative spread has preceded recessions by 12-18 months.")
        st.write("**Recent Volatility:**")
        st.line_chart(df_feat['vol_30d'].tail(20), height=150)

with tab2:
    st.write("### Feature Importance & Inputs")
    st.write("The model analyzes the following features to generate the prediction:")
    cols = st.columns(len(model_package["features"]))
    for i, feature in enumerate(model_package["features"]):
        cols[i].code(feature)
    
    st.divider()
    st.write("### Raw Data Preview")
    st.dataframe(df_feat.tail(10), use_container_width=True)

with tab3:
    st.markdown("""
    ### Why the Yield Curve Matters
    The yield curve is often called the **'Crystal Ball of Capitalism.'** * **Normal Curve:** Long-term rates are higher than short-term rates. This indicates investors expect the economy to grow.
    * **Inverted Curve:** Short-term rates are higher than long-term. This suggests investors are pessimistic about the future and is a classic **recession warning**.
    
    
    """)

st.caption("⚠️ Educational use only. Not financial advice. Data updated as of: " + datetime.now().strftime("%Y-%m-%d"))
