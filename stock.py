# =====================================================
# ADVANCED STOCK MARKET PREDICTION SYSTEM
# Beautiful UI + Higher Accuracy
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #00C9A7;'>
    📈 AI Powered Stock Market Prediction
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------------------------------
# LOAD & TRAIN MODEL
# -----------------------------------------------------
@st.cache_data
def load_and_train():

    df = pd.read_csv(r"C:\Users\Kalyani\OneDrive\Desktop\ml project\NFLX[1].csv")

    # ================= FEATURE ENGINEERING =================
    df['Return'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['Volatility'] = df['Return'].rolling(10).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2

    # Target
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    df.dropna(inplace=True)
    df.drop(['Date', 'Adj Close'], axis=1, inplace=True)

    X = df.drop('Target', axis=1)
    y = df['Target']

    # Time split
    split = int(len(df) * 0.85)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Better Model
    model = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    return model, scaler, acc, df


model, scaler, accuracy, df = load_and_train()

# -----------------------------------------------------
# DASHBOARD LAYOUT
# -----------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Model Accuracy")
    st.metric("Accuracy", f"{accuracy*100:.2f}%")

with col2:
    st.subheader("📈 Dataset Size")
    st.metric("Total Records", len(df))

st.markdown("---")

# -----------------------------------------------------
# STOCK PRICE CHART
# -----------------------------------------------------
st.subheader("📉 Historical Closing Price")

fig = px.line(df, y="Close", title="Stock Price Trend")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------
# SIDEBAR INPUT
# -----------------------------------------------------
st.sidebar.header("🔧 Enter Today's Stock Data")

inputs = []

feature_names = list(df.drop('Target', axis=1).columns)

for feature in feature_names:
    value = st.sidebar.number_input(feature, value=float(df[feature].iloc[-1]))
    inputs.append(value)

# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------
if st.button("🚀 Predict Tomorrow Movement"):

    features = np.array([inputs])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]

    st.subheader("🔮 Prediction Result")

    if prediction == 1:
        st.success(f"📈 STOCK WILL GO UP")
        st.info(f"Confidence: {prob[1]*100:.2f}%")
    else:
        st.error(f"📉 STOCK WILL GO DOWN")
        st.info(f"Confidence: {prob[0]*100:.2f}%")

    st.markdown("---")
    st.subheader("📄 Classification Report")
    st.text(classification_report(df['Target'][-100:], 
                                   model.predict(scaler.transform(
                                   df.drop('Target', axis=1)[-100:]
                                   ))))