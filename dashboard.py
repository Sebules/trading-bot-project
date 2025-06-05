import time

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import yfinance as yf
import os
from alpaca_trade_api.rest import REST
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from broker.alpaca_config import ALPACA_API_KEY, ALPACA_SECRET_KEY

# === CONFIGURATION ===
st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("📊 Dashboard de Trading Algorithmique")

# === SESSION INIT ===
if "data_dict" not in st.session_state:
    st.session_state["data_dict"] = {}

# === CHARGEMENT DE DONNÉES HISTORIQUES ===
st.sidebar.header("🧮 Configuration du portefeuille")

ticker_input = st.sidebar.text_input("📌 Tickers (séparés par des virgules)", value="AAPL, TSLA, MSFT")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
start_date = st.sidebar.date_input("📅 Date de début", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("📅 Date de fin", datetime.date.today())

if st.sidebar.button("📥 Charger les données"):
    st.session_state["data_dict"] = {}
    with st.spinner("Téléchargement en cours..."):
        for ticker in tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(start=start_date, end=end_date)
                time.sleep(1.5)
                if not df.empty:
                    df = df.reset_index()  # Facultatif : Index propre
                    df.set_index("Date", inplace=True)
                    st.session_state["data_dict"][ticker] = df
                    st.success(f"{ticker} récupéré ✅ ({len(df)} lignes)")
                else:
                    st.warning(f"Aucune donnée disponible pour {ticker}")
            except Exception as e:
                st.error(f"Erreur téléchargement {ticker} : {e}")

# === AFFICHAGE D’UN TICKER ===
if st.session_state["data_dict"]:
    st.subheader("🔍 Aperçu des données")
    selected_ticker = st.selectbox("Choisis un ticker à afficher", list(st.session_state["data_dict"].keys()))
    st.dataframe(st.session_state["data_dict"][selected_ticker].head())
else:
    st.info("Aucune donnée chargée. Veuillez entrer les tickers et cliquer sur 'Charger les données'.")

# === CHARGER LES RAPPORTS DE STRATÉGIE ===
st.sidebar.header("⚙️ Paramètres de stratégie")

report_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reporting"))
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

rapport_files = [f for f in os.listdir(report_dir) if f.endswith(".csv")]
if rapport_files:
    selected_rapport = st.sidebar.selectbox("📈 Rapport de stratégie", rapport_files)
    df_report = pd.read_csv(os.path.join(report_dir, selected_rapport), index_col=0)

    if not df_report.empty:
        st.subheader(f"📈 Données pour : {selected_rapport}")
        st.dataframe(df_report.head())

        # === FILTRE DE DATES ===
        try:
            df_report.index = pd.to_datetime(df_report.index)
            start_rpt = st.sidebar.date_input("Date de début (rapport)", df_report.index.min().date())
            end_rpt = st.sidebar.date_input("Date de fin (rapport)", df_report.index.max().date())
            df_filtered = df_report.loc[start_rpt:end_rpt]
        except Exception as e:
            st.error(f"Erreur de date : {e}")
            df_filtered = df_report

        # === GRAPHIQUE AVEC SIGNAUX ===
        st.subheader("📉 Courbe de Performance & Signaux")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df_filtered.index, df_filtered["Close"], label="Prix", alpha=0.5)

        if "Equity_Curve" in df_filtered.columns:
            ax.plot(df_filtered.index, df_filtered["Equity_Curve"], label="Equity Curve", linewidth=2)

        if "Signal" in df_filtered.columns:
            buy = df_filtered[df_filtered["Signal"] == 1]
            sell = df_filtered[df_filtered["Signal"] == 0]
            ax.scatter(buy.index, buy["Close"], marker="^", color="green", label="Buy", alpha=0.8)
            ax.scatter(sell.index, sell["Close"], marker="v", color="red", label="Sell", alpha=0.8)

        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
else:
    st.info("Aucun rapport trouvé dans le dossier 'reporting'.")

# === PORTFOLIO EN TEMPS RÉEL ALPACA ===
st.subheader("📦 Portefeuille en temps réel (Alpaca)")

@st.cache_data
def load_portfolio(api):
    try:
        positions = api.list_positions()
        return pd.DataFrame([p._raw for p in positions])
    except Exception as e:
        st.error(f"Erreur de chargement Alpaca : {e}")
        return pd.DataFrame()

try:
    api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets")
    portfolio_df = load_portfolio(api)

    if not portfolio_df.empty:
        st.dataframe(portfolio_df[["symbol", "qty", "avg_entry_price", "market_value", "unrealized_pl"]])
    else:
        st.warning("Portefeuille vide ou aucune position active.")
except Exception as e:
    st.warning(f"Connexion Alpaca impossible : {e}")