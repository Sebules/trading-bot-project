# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# ====== Chargement des données ======
@st.cache_data
def load_data(filepath):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=True, index_col=0)
        return df
    else:
        st.error(f"Fichier introuvable : {filepath}")
        return None

# ====== Interface utilisateur ======
st.set_page_config(page_title="Dashboard Trading Algo", layout="wide")

st.title("📈 Tableau de bord du Trading Algorithmique")

# Sélection du rapport CSV à afficher
report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reporting"))
if not os.path.exists(report_path):
    st.error(f"❌ Le dossier 'reporting' est introuvable à : {report_path}")
    st.stop()
report_files = [f for f in os.listdir(report_path) if f.endswith(".csv")]
selected_file = st.selectbox("📂 Choisissez un rapport à afficher :", report_files)

# Chargement du fichier sélectionné
df = load_data(os.path.join(report_path, selected_file))

if df is not None:
    st.subheader("📊 Résumé du rapport")
    st.dataframe(df)

    if "Equity_Curve" in df.columns:
        st.subheader("📉 Courbe de performance")
        st.line_chart(df["Equity_Curve"])

    if "Close" in df.columns:
        st.subheader("💰 Prix de l’actif")
        st.line_chart(df["Close"])

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