import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import os
try:
  from alpaca_trade_api.rest import REST, TimeFrame
except Exception:
  REST, TimeFrame = None, None
import io

import sys
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events
try:
  from plotly.offline import iplot, init_notebook_mode
  init_notebook_mode(connected=True)
except Exception:
  iplot, init_notebook_mode = None, None
import ta
from ta.volatility import BollingerBands
from itertools import product
from scipy.optimize import minimize

import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
try:
  from openai import OpenAI
except Exception:
  OpenAI = None
from alpha_vantage.timeseries import TimeSeries
try:
  from alpaca.trading.client import TradingClient
except Exception:
  TradingClient = None
from pathlib import Path


# === CONFIG PATH ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from broker.alpaca_executor import execute_best_signals
from broker.metrics import sharpe_ratio_30j
from strategies.moving_average import MovingAverageStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.breakout_range_strategy import BreakoutRangeStrategy
from strategies.bollinger_bands import BBStrategy
from strategies.ATR_strategy import ATRStrategy
from strategies.SAR_strategy import PSARStrategy
from utils.portfolio import optimize_weights
from utils.compare_strategies import clean_name, compare_strategies
from utils.chat_component import init_chat_with_emilio
from utils.settings import (ALPACA_API_KEY, ALPACA_SECRET_KEY,ALPACA_PAPER_URL, ALPHA_API_KEY,
                             DATA_ROOT, REPORT_ROOT, RESULT_ROOT, ML_TRAIN_ROOT, ML_MODELS_ROOT)
from reporting.performance_report import calculate_performance_metrics, generate_verdict
from execution.execute_trades import get_current_position_qty, execute_signal
from execution.strategies_robot_optimizer import _normalise_returns
from execution.run_bot import load_latest_reports_by_symbol, determine_qty
from ml.train_model import load_report, get_candidate_features, train_model, save_model




# === CONFIGURATION STREAMLIT ===
st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.session_state.setdefault("strategies_to_execute", {})

# === REPERTOIRES ===
DATA_DIR = DATA_ROOT
REPORT_DIR = REPORT_ROOT
RESULT_DIR = RESULT_ROOT
ML_TRAIN_DIR = ML_TRAIN_ROOT
ML_MODELS_DIR = ML_MODELS_ROOT

st.title("📊 Dashboard de Trading Algorithmique")
tabs = st.tabs(["📁 Données", "⚙️ Paramètrages","📊 Graphiques"])

# === SESSION INIT ===
if "data_dict" not in st.session_state:
    st.session_state["data_dict"] = {}

# === OPENAI CHAT -appel du chat Emilio ===
#init_chat_with_emilio()

# === CHARGEMENT DE DONNÉES HISTORIQUES ===
with tabs[0]:
    #try:
        #ALPHA_API_KEY = ALPHA_API_KEY #clé API Alpha VAntage
        #ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')
    #except (KeyError, AttributeError):
        #st.error("🔑 Clé d'API ALPHA VANTAGE non trouvée. Veuillez la configurer dans `.streamlit/secrets.toml`.")
        #st.stop()

    st.header("🧮 Configuration du portefeuille")

    ticker_input = st.text_input("📌 Tickers (séparés par des virgules)", value="AAPL, TSLA, MSFT")
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    start_date = st.date_input("📅 Date de début", datetime(2022, 1, 1))
    end_date = st.date_input("📅 Date de fin", datetime.now().date())

    if st.button("📥 Charger les données"):
        st.session_state["data_dict"] = {}
        with st.spinner("Téléchargement en cours..."):
            for ticker in tickers:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                try:
                    ticker_obj = yf.Ticker(ticker)
                    df = ticker_obj.history(start=start_date, end=end_date)
                    time.sleep(1.5)

                    if not df.empty:
                        df = df[~df.index.duplicated(keep="first")]
                        st.session_state["data_dict"][ticker] = df
                        st.success(f"{ticker} récupéré ✅ ({len(df)} lignes)")
                        df.to_csv(os.path.join(DATA_DIR, f"{ticker}_{timestamp}.csv"))
                        st.success(f"{ticker}_{timestamp} sauvegardé dans data/")
                    else:
                        st.warning(f"Aucune donnée disponible pour {ticker}")
                except Exception as e:
                    st.warning(f"Erreur avec yfinance pour {ticker} : {e}")
                    st.info(f"Tentative avec Alpha Vantage pour {ticker}...")

                    #try:
                        #av_data, meta = ts.get_daily(symbol=ticker, outputsize='full')
                        #start_dt = pd.to_datetime(start_date)
                        #end_dt = pd.to_datetime(end_date)
                        #av_data = av_data[(av_data.index >= start_dt) & (av_data.index <= end_dt)]
                        #av_data = av_data.rename(columns={
                            #'1. open': 'Open',
                            #'2. high': 'High',
                            #'3. low': 'Low',
                            #'4. close': 'Close',
                            #'5. volume': 'Volume'
                        #})

                        #if not av_data.empty:
                            #av_data = av_data[~av_data.index.duplicated(keep="first")]
                            #st.session_state["data_dict"][ticker] = av_data
                            #st.success(f"{ticker} récupéré via Alpha Vantage ✅ ({len(av_data)} lignes)")
                            #av_data.to_csv(os.path.join(DATA_DIR, f"{ticker}_{timestamp}.csv"))
                            #st.success(f"{ticker}_{timestamp} sauvegardé dans data/")
                        #else:
                            #st.warning(f"Aucune donnée pour {ticker} via Alpha Vantage")

                    #except Exception as av_e:
                        #st.error(f"Échec aussi avec Alpha Vantage pour {ticker} : {av_e}")

    # === AFFICHAGE D’UN TICKER ===
    if st.session_state["data_dict"]:
        st.subheader("🔍 Aperçu des données")
        selected_ticker = st.selectbox("Choisis un ticker à afficher", list(st.session_state["data_dict"].keys()))
        st.dataframe(st.session_state["data_dict"][selected_ticker])
    else:
        st.info("Aucune donnée chargée. Veuillez entrer les tickers et cliquer sur 'Charger les données'.")



# === AFFICHAGE D’UN DATAFRAME ===
    st.header("🧮 Affichage d'un Dataframe")
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if data_files:
        selected_df = st.selectbox("📈 Dataframe", data_files)
        df_path = os.path.join(DATA_DIR, selected_df)
        df = pd.read_csv(df_path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_convert(None)
        df = df[~df.index.duplicated(keep="first")]

        df = df.sort_index()

    if not df.empty:
        st.subheader(f"📈 Données pour : {selected_df}")
        st.dataframe(df)

        start_rpt = st.date_input("Date de début (dataframe)", df.index.min().date())
        end_rpt = st.date_input("Date de fin (dataframe)", df.index.max().date())


        if st.button("📅 Appliquer le filtre de dates au dataframe"):
            try:
                start_ts = pd.Timestamp(start_rpt).replace(tzinfo=None)
                end_ts = pd.Timestamp(end_rpt).replace(tzinfo=None)

                df_filtered = df.loc[start_ts:end_ts]

                st.success("Filtrage appliqué avec succès.")
                st.subheader(f"📈 Données filtrées de {start_ts.date()} à {end_ts.date()} pour : {selected_df}")
                st.dataframe(df_filtered)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_name = f"{selected_df}_filtered_{timestamp}.csv"
                df_filtered.to_csv(os.path.join(DATA_DIR, out_name))
                st.success(f"{out_name} sauvegardé dans data/")

            except Exception as e:
                st.error(f"Erreur de date : {e}")
    else:
        st.info("Aucun rapport trouvé dans le dossier 'reporting'.")

    # === PARAMÈTRES DE STRATÉGIES MULTIPLES ===
    st.header("⚙️ Application de stratégie")
    # === 📝 Affichage du mémo Top 3 (si présent) ===
    top3_configs = st.session_state.get("top3_configs", {})
    if top3_configs:
        st.subheader("📋 Mémo – Meilleures configurations par stratégie")
        # Pour chaque ticker mémorisé, on affiche son Top 3
        for ticker, df_top3 in top3_configs.items():
            st.markdown(f"**{ticker}**")
            st.dataframe(df_top3)
    if top3_configs:
        # → Préparation du CSV combiné
        memo_df = pd.concat(
            [df.assign(Ticker=ticker) for ticker, df in top3_configs.items()],
            ignore_index=True
        )
        csv_bytes = memo_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="📥 Télécharger le mémo au format CSV",
            data=csv_bytes,
            file_name="memo_top3.csv",
            mime="text/csv"
        )


    strategy_options = {
        "Moving Average": {"class": MovingAverageStrategy, "params": ["short_window", "long_window"]},
        "RSI": {"class": RSIStrategy, "params": ["window", "rsi_low", "rsi_high"]},
        "Breakout Range": {"class": BreakoutRangeStrategy, "params": ["lookback"]},
        "Bollinger Bands":{"class": BBStrategy, "params":["window_bb", "window_dev_bb" ]},
        "ATR":{"class": ATRStrategy, "params":"window_ATR"},
        "PSAR":{"class":PSARStrategy}
    }
    # Valeurs par défaut spécifiques par stratégie
    default_values = {
        "RSI": {"window": 14, "rsi_low": 30, "rsi_high": 70},
        "Moving Average": {"short_window": 10, "long_window": 30},
        "Breakout Range": {"lookback": 20},
        "Bollinger Bands": {"window_bb": 20, "window_dev_bb":2 },
        "ATR":{"window_ATR":14}
    }

    selected_strategies = st.multiselect("📌 Choisis les stratégies à appliquer", list(strategy_options.keys()),key="selected_strategies")

    strategy_params = {}
    for idx,name in enumerate(selected_strategies):
        with st.expander(f"🔧 Paramètres : {name}"):
            defaults = default_values.get(name, {})
            params = {}
            for jdx,(param,default) in enumerate(defaults.items()):
                key = f"param_{name}_{param}_{idx}_{jdx}"
                #default = default_values.get(name, {}).get(param, 10)
                params[param] = st.number_input(param, min_value=1, value=default, step=1,key=key)
            strategy_params[name] = params

    # === Sélection des fichiers à traiter ===
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    selected_files = st.multiselect(
        "📂 Choisis les fichiers pour l'application des stratégies",
        options=data_files,
        default=data_files,
        help="Par défaut toutes les données sont sélectionnées, décochez pour cibler un seul fichier."
    )

    if st.button("⚙️ Appliquer les stratégies sélectionnées"):
        st.subheader("📄 Résumé de la génération des rapports")
        if not selected_files:
            st.warning("⚠️ Aucun fichier sélectionné.")
        else:
            for file in selected_files:
                file_path = os.path.join(DATA_DIR, file)
                symbol = file.replace(".csv", "")
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    df = df[~df.index.duplicated(keep="first")]
                    if "Close" not in df.columns:
                        st.warning(f"❌ {file} ignoré (pas de colonne 'Close')")
                        continue

                    mkt = df["Close"].pct_change().fillna(0)
                    df_global = df.copy()
                    for strat_name in selected_strategies:
                        strat_class = strategy_options[strat_name]["class"]
                        params = strategy_params[strat_name]
                        strat = strat_class(df.copy(), **params)
                        result_df = strat.generate_signals()

                        if strat_name not in ("ATR","PSAR"):

                            strategy_returns = result_df["Signal"].shift(1).fillna(0) * mkt

                            df_global[f"{strat_name}_Returns"] = strategy_returns
                            df_global[f"{strat_name}_Equity"] = (1 + strategy_returns).cumprod()
                            df_global[f"{strat_name}_Signal"] = result_df["Signal"]


                        # === 🆕 Sauvegarde des colonnes techniques ===
                        for col in result_df.columns:
                            if col not in df_global.columns and col not in ["Signal", "Returns", "Position", "Strategy"]:
                                df_global[col] = result_df[col]

                    # Ajout de l'horodatage
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    report_name = f"rapport_{symbol}_STGY_{timestamp}.csv"
                    df_global.to_csv(os.path.join(REPORT_DIR, report_name))
                    st.success(f"✅ Rapport généré : {report_name}")
                except Exception as e:
                                st.error(f"❌ Erreur sur {file} avec {strat_name} : {e}")

    # === CHARGER LES RAPPORTS DE STRATÉGIE ===
    st.header("⚙️ Paramètres de stratégie")

    rapport_files = [f for f in os.listdir(REPORT_DIR) if f.endswith(".csv")]
    if rapport_files:
        selected_rapport = st.selectbox("📈 Rapport de stratégie", rapport_files)
        rapport_path = os.path.join(REPORT_DIR, selected_rapport)
        df_report = pd.read_csv(rapport_path, index_col=0)
        df_report.index = pd.to_datetime(df_report.index, utc=True)
        df_report.index = df_report.index.tz_convert(None)
        df_report = df_report[~df_report.index.duplicated(keep="first")]

        if not df_report.empty:
            st.subheader(f"📈 Données pour : {selected_rapport}")
            st.dataframe(df_report)

            start_rpt = st.date_input("Date de début (rapport)", df_report.index.min().date())
            end_rpt = st.date_input("Date de fin (rapport)", df_report.index.max().date())


            if st.button("📅 Appliquer le filtre de dates"):
                try:
                    start_rpt = pd.Timestamp(start_rpt).replace(tzinfo=None)
                    end_rpt = pd.Timestamp(end_rpt).replace(tzinfo=None)
                    df_filtered = df_report.loc[start_rpt:end_rpt]
                    st.success("Filtrage appliqué avec succès.")
                    st.subheader(f"📈 Données filtrées de {start_rpt.date()} à {end_rpt.date()} pour : {selected_rapport}")
                    st.dataframe(df_filtered)

                except Exception as e:
                    st.error(f"Erreur de date : {e}")
    else:
        st.info("Aucun rapport trouvé dans le dossier 'reporting'.")



# === UTILITAIRE POUR COLORATION DE LA DECISION ===

def color_decision(val):
    if "CONSERVER" in val:
        return "background-color: lightgreen"
    elif "ÉVITER" in val:
        return "background-color: salmon"
    elif "AMÉLIORER" in val:
        return "background-color: orange"
    return ""

# === COMPARAISON DES STRATÉGIES DANS UN MÊME RAPPORT ===
with tabs[1]:
    st.subheader("📊 Comparaison des stratégies d’un même rapport")

    rapport_compare = st.selectbox("📁 Choisir un rapport contenant plusieurs stratégies",
                                   [f for f in os.listdir(REPORT_DIR) if f.endswith(".csv")])

    if "df_perf_comparaison" not in st.session_state:
        st.session_state["df_perf_comparaison"] = None
        st.session_state["equity_cols"] = []
        st.session_state["df_comparaison_raw"] = None

    if st.button("🧪 Lancer la comparaison sur ce rapport"):
        try:
            path = os.path.join(REPORT_DIR, rapport_compare)
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

            # Détection automatique des colonnes Equity
            equity_cols = [col for col in df.columns if col.endswith("_Equity")]
            return_cols = [col.replace("_Equity", "_Returns") for col in equity_cols]

            perf_results = {}
            for equity, ret in zip(equity_cols, return_cols):
                strat_name = equity.replace("_Equity", "")
                signal_col = f"{strat_name}_Signal"

                if ret in df.columns and signal_col in df.columns:
                    df_temp = pd.DataFrame({
                        "Strategy_Returns": df[ret].fillna(0),
                        "Signal": df[signal_col]
                    })


                    perf = calculate_performance_metrics(df_temp, strategy_col="Strategy_Returns")
                    # Ajoute le Sharpe 30j harmonisé
                    sr30 = sharpe_ratio_30j(df, ret)
                    perf["Sharpe 30j"] = float("nan") if sr30 is None else sr30



                    decision, verdict, score = generate_verdict(perf)
                    perf["Decision"] = decision
                    perf["Verdict"] = ", ".join(verdict)
                    perf["Score"] = score

                    perf_results[strat_name] = perf

            if perf_results:
                df_perf = pd.DataFrame(perf_results).T.sort_values("Score", ascending=False)

                st.success("✅ Résultats de comparaison générés")
                #st.dataframe(df_perf)
                styled = (
                    df_perf.style
                    .applymap(color_decision, subset=["Decision"])
                    .format("{:.2f}", subset=["Total Return (%)", "Sharpe Ratio","Sharpe 30j", "Max Drawdown (%)",
                                              "Win Rate (%)", "Number of Trades", "Profit Factor", "Volatilité (%)",
                                              "Calmar Ratio"])
                    .set_table_styles([
                        {"selector": "th", "props": [("font-size", "12px"), ("text-align", "center")]},
                        {"selector": "td", "props": [("font-size", "12px"), ("padding", "5px")]}
                    ])
                    .set_properties(**{
                        "text-align": "left",
                        "white-space": "pre-wrap",
                        "word-wrap": "break-word",
                        "max-width": "180px"
                    })
                )

                # ✅ Injecter un style global dans la page pour désactiver le troncage des colonnes
                custom_css = """
                <style>
                table {
                    table-layout: auto !important;
                    width: 100% !important;
                }
                thead th {
                    text-align: center !important;
                }
                td {
                    word-wrap: break-word !important;
                    white-space: pre-wrap !important;
                    max-width: 180px !important;
                }
                </style>
                """

                st.markdown(custom_css, unsafe_allow_html=True)
                st.markdown(styled.to_html(), unsafe_allow_html=True)

                st.subheader("🧾 Recommandations d'action par stratégie")

                # Tri basé sur le DataFrame de performance
                action_map = {
                    1: "🟢 Acheter",
                    -1: "🔴 Vendre",
                    0: "🟡 Conserver"
                }

                # On utilise les noms de stratégies dans l'ordre du classement
                for strat_name in df_perf.index:
                    score = df_perf.loc[strat_name, "Score"]
                    if score <= 0:
                        st.markdown(f"**{strat_name}** ➤ ⛔ Ignorée (stratégie non fiable)")
                        continue

                    return_col = f"{strat_name}_Returns"
                    signal_col = f"{strat_name}_Signal" if f"{strat_name}_Signal" in df.columns else "Signal"

                    try:
                        if signal_col in df.columns:
                            last_signal = int(df[signal_col].dropna().iloc[-1])
                        elif return_col in df.columns:
                            last_return = df[return_col].dropna().iloc[-1]
                            last_signal = int(np.sign(last_return))
                        else:
                            last_signal = 0  # Par défaut
                    except Exception as e:
                        last_signal = 0

                    action = action_map.get(last_signal, "❔ Inconnu")
                    st.markdown(f"**{strat_name}** ➤ {action}")

                # Graphe d'évolution
                st.write("📉 Courbe Equity comparée")
                fig, ax = plt.subplots(figsize=(14, 6))

                for col in equity_cols:
                    ax.plot(df.index, df[col], label=col.replace("_Equity", ""))
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                st.session_state["df_perf_comparaison"] = df_perf
                st.session_state["equity_cols"] = equity_cols
                st.session_state["df_comparaison_raw"] = df
            else:
                st.warning("⚠️ Aucune stratégie valide détectée pour la comparaison.")

        except Exception as e:
            st.error(f"❌ Erreur lors de la comparaison : {e}")


    # === BOUTON TELECHARGEMENT FICHIER ===
    if st.session_state["df_perf_comparaison"] is not None:

        if st.button("💾 Télécharger le rapport comparatif"):
            df_perf = st.session_state["df_perf_comparaison"]
            df = st.session_state["df_comparaison_raw"]

            try:
                os.makedirs(RESULT_DIR, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path_csv = os.path.join(RESULT_DIR, f"resultat_{timestamp}.csv")
                file_path_xlsx = os.path.join(RESULT_DIR, f"resultat_{timestamp}.xlsx")

                df_perf.to_csv(file_path_csv)

                styled = df_perf.style.applymap(color_decision, subset=["Decision"])
                with pd.ExcelWriter(file_path_xlsx) as writer:
                    styled.to_excel(writer, sheet_name="Comparaison")

                st.success(f"✅ Rapport enregistré dans :\n{file_path_csv}\n{file_path_xlsx}")
            except Exception as e:
                st.error("❌ Erreur lors de la sauvegarde")
                st.exception(e)


# === GRAPHIQUE INTERACTIF PLOTLY ===
with tabs[2]:
    # On vérifie que les DataFrames existent dans la session
    if st.session_state.get("df_perf_comparaison") is not None:

        # Récupération des données
        df_perf = st.session_state["df_perf_comparaison"]
        df = st.session_state["df_comparaison_raw"]
        equity_cols = st.session_state["equity_cols"]


        st.subheader("📊 Graphe interactif des stratégies")

        try:
            # --- Sélections utilisateur ---
            strat_options = [col.replace("_Equity", "") for col in equity_cols]
            selected_plot_strats = st.multiselect("📈 Stratégies à afficher", strat_options, default=strat_options)

            technical_candidates = [col for col in df.columns if
                                    col.lower() in ["rsi", "sma_short", "sma_long", "bb_lower", "bb_upper", "bb_middle",
                                                    "breakout_haut", "breakout_bas","atr","stop_loss_atr_l",
                                                    "stop_loss_atr_s", "take_profit_atr_l","take_profit_atr_s",
                                                    "stop_loss"]]

            selected_techs = st.multiselect("🧮 Indicateurs techniques à afficher", technical_candidates)

            # --- Création de la figure
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])


            # 1. Chandeliers
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                         name="Chandelier"), row=1, col=1)

            # 2. Volume coloré
            if "Volume" in df.columns:
                colors = np.where(df['Close'] >= df['Open'], 'green', 'red')
                fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors), row=2, col=1)

            # 3. Indicateurs techniques
            for tech in selected_techs:
                if tech !="stop_loss":
                    fig.add_trace(go.Scatter(x=df.index, y=df[tech], mode="lines", name=tech), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=df.index, y=df[tech], mode="markers", name=tech,
                                             marker=dict(color="pink", symbol="asterisk", size=8)), row=1, col=1)

            # 4. Stratégies
            for strat in selected_plot_strats:

                equity_col = f"{strat}_Equity"
                signal_col = f"{strat}_Signal" if f"{strat}_Signal" in df.columns else "Signal"
                if equity_col in df.columns: fig.add_trace(
                    go.Scatter(x=df.index, y=df[equity_col], mode="lines", name=f"{strat} Equity"), row=1, col=1)
                if signal_col in df.columns:
                    buy_signals = df[df[signal_col] == 1]
                    sell_signals = df[df[signal_col] == -1]
                    fig.add_trace(
                        go.Scatter(x=buy_signals.index, y=buy_signals["Close"], mode="markers", name=f"{strat} Buy",
                                   marker=dict(color="green", symbol="triangle-up", size=8)), row=1, col=1)
                    fig.add_trace(
                        go.Scatter(x=sell_signals.index, y=sell_signals["Close"], mode="markers", name=f"{strat} Sell",
                                   marker=dict(color="red", symbol="triangle-down", size=8)), row=1, col=1)



            # --- Mise en page finale ---
            fig.update_layout(
                title="Évolution des stratégies et signaux",
                template="plotly_white",
                height=800,
                xaxis_rangeslider_visible=False,

                # Active le mode "survol" qui affiche les infos de toutes les courbes pour une même date
                hovermode='x unified'
            )

            # Configure l'apparence des lignes qui suivent le curseur
            fig.update_xaxes(
                showspikes=True,  # Montre la ligne verticale
                spikemode='across',  # La ligne traverse tous les sous-graphiques
                spikesnap='cursor',  # La ligne suit précisément le curseur
                spikedash='dot',  # Style de la ligne (pointillés)
                spikecolor='grey',  # Couleur de la ligne
                spikethickness=1  # Épaisseur de la ligne
            )

            fig.update_yaxes(
                showspikes=True,  # Montre la ligne horizontale
                spikedash='dot',
                spikecolor='grey',
                spikethickness=1
            )

            fig.update_traces(selector=dict(type='bar'), showlegend=False)

            # Affichage
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error("❌ Une erreur est survenue lors de la création du graphique.")
            st.exception(e)



