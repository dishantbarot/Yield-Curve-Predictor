# 📈 Yield Curve Predictor  
### Machine Learning–Based Forecasting of U.S. Treasury Yield Curve Dynamics

---

## 📌 Project Overview

The **Yield Curve Predictor** is a machine learning–driven financial analytics project designed to model, predict, and interpret movements in the U.S. Treasury yield curve.

The yield curve encapsulates market expectations about **economic growth, inflation, and monetary policy**, making it a critical tool for fixed-income investors, policymakers, and macroeconomic analysts.

This project integrates **real-time financial data**, **finance-driven feature engineering**, **supervised machine learning**, and **interactive visualization** to deliver both **quantitative predictions** and **economic insights**.

---

## 🎯 Objectives

- Fetch real-time U.S. Treasury yield data (1-month to 30-year maturities)
- Engineer economically meaningful yield-curve features
- Train and evaluate machine learning models for interest-rate forecasting
- Predict the next-period **10-Year Treasury yield**
- Visualize yield curve dynamics and inversions
- Explain the economic significance of yield movements
- Deploy the solution as an interactive **Streamlit application**

---

## 🧠 Why the Yield Curve Matters

The yield curve represents the relationship between **interest rates and time to maturity**.

It provides insights into:
- Economic expansion or slowdown
- Inflation expectations
- Central bank policy stance
- Recession risk

### Yield Curve Inversion
When short-term rates (e.g., 2Y) exceed long-term rates (e.g., 10Y), the curve becomes **inverted**.  
Historically, yield curve inversions have preceded most U.S. recessions, making them one of the most reliable macroeconomic indicators.

---

## 🏗️ Project Architecture


---

## 📊 Data Acquisition

**Source:** Federal Reserve Economic Data (FRED)  
**Library Used:** `fredapi`

### Treasury Maturities Used
- 1 Month (1M)
- 3 Month (3M)
- 1 Year (1Y)
- 2 Year (2Y)
- 5 Year (5Y)
- 10 Year (10Y)
- 30 Year (30Y)

The data is:
- Time-indexed
- Aligned across maturities
- Cleaned for missing values
- Cached to minimize API calls

---

## 🧮 Feature Engineering (Core Intelligence Layer)

Feature engineering is driven by fixed-income theory and macroeconomic intuition.

### 1️⃣ Yield Spreads
Capture yield curve shape and recession signals:
- 10Y − 2Y
- 5Y − 3M
- 30Y − 10Y

### 2️⃣ Rolling Statistics
Capture market regimes:
- 30-day and 90-day volatility
- 30-day and 90-day moving averages

### 3️⃣ Lag Features
Capture market memory and autocorrelation:
- 1-day, 5-day, and 10-day lag of the 10Y yield

All features are created **before a final NaN cleanup** to preserve time-series integrity.

---

## 🎯 Target Variable

**Prediction Task:**  
Forecast the **next-period 10-Year Treasury yield** using only information available at the current time.

This ensures:
- No data leakage
- Causal and realistic forecasting
- Production-ready modeling

---

## 🤖 Machine Learning Models

### Baseline Model
- **Linear Regression**
- Used for interpretability and benchmarking

### Final Model (Deployed)
- **Random Forest Regressor**
- Captures non-linear relationships and regime shifts
- Robust to noise and macroeconomic transitions

The deployed model is stored as a **package** containing:
- Trained model object
- Feature column schema (for deployment safety)

---

## 📈 Model Evaluation

Evaluation metrics:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score**

Models are evaluated using **time-aware train/test splits** to simulate real-world forecasting conditions.

---

## 🖥 Streamlit Application

### Key Features

#### 📊 Yield Curve Visualization
- Interactive Plotly chart comparing **2-Year vs 10-Year yields**
- Highlights curve flattening and inversion periods

#### 🔮 Model Prediction
- Displays predicted next-period 10Y yield

#### 🧪 Scenario Analysis
- User-controlled yield shock (basis points)
- Demonstrates sensitivity to macroeconomic changes

#### 📌 Feature Importance
- Visualizes the most influential drivers behind predictions

#### 📘 Economic Interpretation
- Explains:
  - Yield curve significance
  - Inversion signals
  - Impact on mortgages, corporate borrowing, savings, and inflation

---

## 🔐 Security & Best Practices

- API keys stored securely using Streamlit `secrets.toml`
- Cached API calls for efficiency
- Modular and readable code structure
- Defensive checks for missing data and schema mismatches

---

## 🧠 Key Insights

- Yield curve slope and long-end rates dominate predictive power
- Autocorrelation alone can inflate accuracy metrics if not controlled
- Finance-driven feature engineering is critical for robust forecasting
- Machine learning complements, rather than replaces, economic theory

---

## 🚀 Future Enhancements

- Yield curve regime classification (normal / flat / inverted)
- Recession probability modeling
- Time-series cross-validation
- SHAP-based explainability
- Cloud deployment with Streamlit Community Cloud

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
It does not constitute financial advice.

---

## 👤 Author

**Dishant Barot**  
Data Scientist | FinTech & Quantitative Finance Enthusiast  

---

⭐ If you found this project insightful, feel free to star the repository!
