import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from alpaca_trade_api.rest import REST

# === CONFIG ===
st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("📊 Dashboard de Trading Algorithmique")

# === CHARGEMENT DE DONNÉES ===
def load_data(filepath):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=True, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()

@st.cache_data
def load_portfolio(api):
    try:
        positions = api.list_positions()
        return pd.DataFrame([p._raw for p in positions])
    except Exception as e:
        st.error(f"Erreur chargement portefeuille : {e}")
        return pd.DataFrame()

# === FILTRE PAR STRATÉGIE ===
st.sidebar.header("⚙️ Paramètres")
report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reporting"))
if not os.path.exists(report_path):
    st.error(f"❌ Le dossier 'reporting' est introuvable à : {report_path}")
    st.stop()
rapport_files = [f for f in os.listdir(report_path) if f.endswith(".csv")]
selected_rapport = st.sidebar.selectbox("📈 Choisis une stratégie", rapport_files)

df = load_data(os.path.join(report_path, selected_rapport))

if not df.empty:
    st.subheader(f"📈 Données pour : {selected_rapport}")
    st.dataframe(df.head())

    # === FILTRE DE DATES ===
    start_date = st.sidebar.date_input("Date de début", df.index.min().date())
    end_date = st.sidebar.date_input("Date de fin", df.index.max().date())

    df_filtered = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    # === GRAPHIQUE AVEC SIGNAUX ===
    st.subheader("📉 Courbe de Performance & Signaux")
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df_filtered.index, df_filtered["Close"], label="Prix", alpha=0.5)
    if "Equity_Curve" in df_filtered.columns:
        ax.plot(df_filtered.index, df_filtered["Equity_Curve"], label="Equity Curve", linewidth=2)

    if "Signal" in df_filtered.columns:
        buy_signals = df_filtered[df_filtered["Signal"] == 1]
        sell_signals = df_filtered[df_filtered["Signal"] == 0]

        ax.scatter(buy_signals.index, buy_signals["Close"], marker="^", color="green", label="Buy", alpha=0.9)
        ax.scatter(sell_signals.index, sell_signals["Close"], marker="v", color="red", label="Sell", alpha=0.9)

    ax.legend()
    ax.set_title("Stratégie avec Signaux Buy/Sell")
    ax.grid(True)
    st.pyplot(fig)

# === PORTFOLIO ALPACA ===
#import sys
# === Ajout manuel du dossier parent pour trouver le dossier broker ===
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from broker.alpaca_config import ALPACA_API_KEY,ALPACA_SECRET_KEY
try:
    ALPACA_API_KEY = st.secrets["PK61T615PP7B5CRY9SZJ"]
    ALPACA_SECRET_KEY = st.secrets["wHUSWtFra5V9XpWLxOReZoK5wzsFcyCDpOjlFPpU"]
    api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets")

    st.subheader("📦 Portefeuille en temps réel (Alpaca)")
    portfolio_df = load_portfolio(api)
    if not portfolio_df.empty:
        st.dataframe(portfolio_df[["symbol", "qty", "avg_entry_price", "market_value", "unrealized_pl"]])
except Exception as e:
    st.warning("Clés API Alpaca non configurées ou connexion impossible.")
