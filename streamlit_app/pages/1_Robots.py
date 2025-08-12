import pandas as pd
import numpy as np
import streamlit as st
import os
import sys
import time
from pathlib import Path
import ta
from ta.volatility import BollingerBands
from itertools import product
from scipy.optimize import minimize
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
try:
  from alpaca_trade_api.rest import REST
except Exception:
  REST = None

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root)
from execution.select_by_prediction import predict_top_n
from execution.risk_clustering import (cluster_and_select, get_tradeable_symbols, prepare_cluster_stats_df,
                                       build_cluster_scatter)
from strategies.moving_average import MovingAverageStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.breakout_range_strategy import BreakoutRangeStrategy
from strategies.bollinger_bands import BBStrategy
from strategies.ATR_strategy import ATRStrategy
from strategies.SAR_strategy import PSARStrategy
from reporting.performance_report import calculate_performance_metrics
from utils.portfolio import optimize_weights, generate_random_portfolios
from utils.chat_component import init_chat_with_emilio
from utils.compare_strategies import clean_name, compare_strategies
from utils.settings import (ALPACA_API_KEY, ALPACA_SECRET_KEY,ALPACA_PAPER_URL,
                             DATA_ROOT, REPORT_ROOT, RESULT_ROOT, ML_TRAIN_ROOT, ML_MODELS_ROOT)
from ml.train_model import load_report,load_model, get_candidate_features, train_model, save_model
from execution.run_bot import (execute_best_signals, load_latest_reports_by_symbol, determine_qty,
                               load_latest_data_by_symbol)

from execution.strategies_robot_optimizer import run_strategy_optimizer
from broker.persistence import (save_best_strat_json, save_best_strat_sqlite,
                                load_best_strat_json, upsert_best_strat_sqlite, load_best_strat_sqlite)
from broker.metrics import sharpe_ratio_30j




# === CONFIG PATH ===
DATA_DIR = DATA_ROOT
REPORT_DIR = REPORT_ROOT
RESULT_DIR = RESULT_ROOT
ML_TRAIN_DIR = ML_TRAIN_ROOT
ML_MODELS_DIR = ML_MODELS_ROOT

try:
  api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_PAPER_URL)
except Exception:
  api = None

# === OPENAI CHAT -appel du chat Emilio ===
#init_chat_with_emilio()
st.title("🤖 Robots pour Trading Algorithmique")
tabs = st.tabs(["🏗️ ML Strategies", "⚙️ Prédictions & Risques","📊 Selection de Strategies"])

# === CREATION MODELE MACHINE LEARNING ===
with tabs[0]:
    st.subheader("🏗️ Création d'un modèle Machine Learning à partir d’un rapport")

    rapport_ml = st.selectbox("📁 Choisir un rapport avec stratégies", [f for f in os.listdir(REPORT_DIR) if f.endswith(".csv")])

    if st.button("🔍 Charger et préparer le rapport"):
        df = load_report(REPORT_DIR, rapport_ml)
        if df is not None:
            st.session_state["df_ml"] = df
            st.success("✅ Rapport chargé et préparé.")
            st.write(df.head())
        else:
            st.error("❌ Erreur lors du chargement du rapport.")

    if "df_ml" in st.session_state:
        df = st.session_state["df_ml"]
        features = get_candidate_features(df)
        selected_features = st.multiselect("📊 Sélectionne les colonnes de features", features, default=features)

        model_type = st.selectbox("🧠 Choix du modèle", ["RandomForestClassifier", "SVC"])

        params = {}
        if model_type == "RandomForestClassifier":
            params["n_estimators"] = st.number_input("🌲 n_estimators", min_value=10, max_value=500, value=100, step=10)
            params["random_state"] = st.number_input("🎲 random_state", min_value=0, max_value=9999, value=42, step=1)

        # On ajoute une section pour que l'utilisateur spécifie les paramètres des stratégies
        with st.expander("🔬 Spécifier les paramètres des stratégies utilisées pour les features"):
            strategy_params_for_saving = {
                "Moving Average": {
                    "short_window": st.number_input("SMA Short Window", 10),
                    "long_window": st.number_input("SMA Long Window", 30)},

                "RSI": {"window": st.number_input("RSI Window", 14),
                        "rsi_low":st.number_input("RSI low", 30),
                        "rsi_high":st.number_input("RSI high", 70)},

                "Bollinger Bands": {"window_bb": st.number_input("BB Window", 20),
                                    "window_dev_bb": st.number_input("BB Deviation", 2)},

                "Breakout Range": {"lookback": st.number_input("Breakout Lookback", 20)}
            }

        if st.button("🏁 Entraîner le modèle"):
            if not selected_features:
                st.warning("⚠️ Veuillez sélectionner au moins une colonne de features.")
            else:
                try:
                    model, model_params, accuracy, report = train_model(df, selected_features, model_type, **params)

                    st.success(f"✅ Entraînement terminé - Accuracy : {accuracy:.2f}")
                    st.text(classification_report(df['Target'][-len(report):], model.predict(df[selected_features][-len(report):])))

                    # Sauvegarde
                    model_name, saved_accuracy= save_model(model=model,
                                                           model_params=params,
                                                           report=report,
                                                           selected_features=selected_features,
                                                          rapport_filename=rapport_ml,
                                                           model_type=model_type,
                                                           model_dir=ML_MODELS_DIR,
                                                           accuracy=accuracy,
                                                          strategy_params=strategy_params_for_saving
                                                           )
                    st.success(f"💾 Modèle enregistré sous : `{model_name}`(Accuracy: {saved_accuracy:.2f})")

                except Exception as e:
                    st.error(f"❌ Erreur d'entraînement : {e}")

# === PREDICTIONS ET RISQUES ===
with tabs[1]:
    # ─── ➊ SECTION : Sélection par prédiction ─────────────────────────
    st.header("🤖 Sélection par prédiction de mouvement")
    if REST is None or TradingClient is None:
          st.warning("Mode démo : les fonctionnalités Alpaca sont désactivées sur Streamlit Cloud.")
          st.info("Pour les ordres et le bot, utilisez l’environnement local / script auto_bot.py.")
          st.stop()  # ou return si c’est dans une fonction/page
    st.subheader("🔄 Charger un modèle pré-entrainé")

    model_files = [f for f in os.listdir(ML_MODELS_DIR) if f.endswith(".json")]

    if not model_files:
        st.warning("Aucun fichier de modèle (.json) trouvé dans le dossier /ml/trained_models.")
    else:
        model_filename = st.selectbox("Choisir le fichier modèle", os.listdir(ML_MODELS_DIR))
        model_path = os.path.join(ML_MODELS_DIR, model_filename)
        try:
            model_dir, filename = os.path.split(model_path)
            model_name, _ = os.path.splitext(filename)
            model, metadata = load_model(model_dir, model_name)
            st.success(f"✅ Modèle « {model_name} » chargé depuis : {model_dir}")

            # On utilise la liste de features spécifique au modèle chargé, et non une liste générique.
            features_list = metadata['features_used']
            n = st.slider("Nombre de titres à cibler", min_value=5, max_value=50, value=20, step=5)
            if st.button("▶️ Lancer la prédiction"):
                top = predict_top_n(model, features_list, n, metadata)
                df_pred = pd.DataFrame(top, columns=["Ticker", "Probabilité Hausse"])
                st.session_state["df_pred"] = df_pred

        except Exception as e:
            st.error(f"❌ Échec du chargement ou de l'exécution du modèle : {e}")

    if "df_pred" in st.session_state:
        st.subheader("📋 Résultats de la prédiction")
        st.dataframe(st.session_state["df_pred"], use_container_width=True)

    # ─── ➋ SECTION : Sélection par risque ────────────────────────────
    st.header("🛡️ Sélection par facteurs de risque")
    if REST is None or TradingClient is None:
          st.warning("Mode démo : les fonctionnalités Alpaca sont désactivées sur Streamlit Cloud.")
          st.info("Pour les ordres et le bot, utilisez l’environnement local / script auto_bot.py.")
          st.stop()  # ou return si c’est dans une fonction/page
    try:
      symbols = get_tradeable_symbols(api)
    except Exception:
      symbols = None
    k = st.number_input("Nombre de clusters", min_value=2, max_value=10, value=5)

    if st.button("▶️ Lancer le clustering"):
        selected, clusters, stats = cluster_and_select(symbols, n_clusters=k, top_k=1)

        # ─── Stockage de la sélection courante ───
        st.session_state["cluster_sel"] = selected
        st.session_state["cluster_stats"] = stats
        st.session_state["cluster_dict"] = clusters
        st.session_state["cluster_k"] = k

    if "cluster_stats" in st.session_state:
        stats = st.session_state["cluster_stats"].copy()
        selected = st.session_state["cluster_sel"]
        clusters = st.session_state["cluster_dict"]
        k_clusters = st.session_state["cluster_k"]

        # ─── Construction du DataFrame pour Plotly ───
        stats_df = prepare_cluster_stats_df(stats, selected)

        search_ticker = st.text_input("🔎 Rechercher un ticker").upper().strip() or None

        fig = build_cluster_scatter(
            stats=stats_df,
            k_clusters=k_clusters,  # le nombre de clusters choisi
            search=search_ticker,  # None si champ vide
        )

        st.plotly_chart(fig, use_container_width=True)
        st.success(f"✅ Tickers sélectionnés (1 par cluster) : {selected}")
        for c, tick_list in clusters.items():
            st.write(f"Cluster {c} ({len(tick_list)} titres)")


# === SECTION ROBOT DE SÉLECTION DES STRATÉGIES ===
with tabs[2]:
    st.header("🤖 Robot d’optimisation des stratégies")

    # 1️⃣ Sélection du ticker / DataFrame
    data_files_map = {sym: path for sym, path in load_latest_data_by_symbol().items()}
    tickers = list(data_files_map.keys())

    selected_tickers = st.multiselect("📂 Choisir un fichier de données", tickers)

    # 2️⃣ Choix des stratégies à tester
    strategy_choices = {
        "Moving Average": MovingAverageStrategy,
        "RSI": RSIStrategy,
        "Breakout Range": BreakoutRangeStrategy,
        "Bollinger Bands": BBStrategy
    }
    selected_strategies = st.multiselect("Stratégies à optimiser", list(strategy_choices.keys()),
                                         default=list(strategy_choices.keys()))

    # 3️⃣ Paramètres grids prédéfinis
    param_grid = {
        "Moving Average": {
            "short_window": range(5, 21),
            "long_window": range(10, 101)
        },
        "RSI": {
            "window": range(5, 31),
            "rsi_low": range(10, 51),
            "rsi_high": range(50, 91)
        },
        "Breakout Range": {
            "lookback": range(10, 51)
        },
        "Bollinger Bands": {
            "window_bb": range(10, 41),
            "window_dev_bb": range(1, 6)
        }
    }

    with st.form("form_strategy_search"):
        lancer_robot = st.form_submit_button("▶️ Lancer le Robot de stratégie")

    if lancer_robot:
        run_strategy_optimizer(
            selected_tickers=selected_tickers,
            selected_strategies=selected_strategies,
            param_grid=param_grid,
            data_files_map=data_files_map,
            strategy_choices=strategy_choices,
            data_dir=DATA_DIR,  # ou un autre dossier si besoin
        )


    st.header("🤖 Optimisation de portefeuille multi-stratégies")

    # 1. Chargement des rapports disponibles
    reports = {sym: path for sym, path in load_latest_reports_by_symbol().items()}
    tickers = list(reports.keys())
    strategies = selected_strategies  # celles que l'utilisateur a déjà choisies

    # 2. Permettre à l'utilisateur de choisir les tickers à inclure
    selected_tickers_portfolio = st.multiselect(
        "📂 Sélectionnez les tickers à optimiser dans le portefeuille",
        options=tickers,
        default=tickers
    )

    with st.form("form_portfolio_opt"):
        lancer_opt = st.form_submit_button("▶️ Lancer l’optimisation portefeuille")
    if lancer_opt:
        if not selected_tickers_portfolio:
            st.warning("⚠️ Veuillez sélectionner au moins un ticker pour l’optimisation.")
            st.stop()
        # 3. Construction de returns_df filtré sur la sélection
        rets = {}
        for sym in selected_tickers_portfolio:
            path = reports[sym]
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df[~df.index.duplicated(keep="first")]
            for strat in strategies:
                col = f"{strat}_Returns"
                if col in df.columns:
                    rets[f"{sym}_{strat}"] = df[col].fillna(0)
        returns_df = pd.DataFrame(rets).dropna(how="all").fillna(0)


        # 4. Affichage harmonisé du Sharpe 30 jours pour chaque combinaison
        sr_rows = []
        if not returns_df.empty:
            for col in returns_df.columns:
                sr = sharpe_ratio_30j(returns_df, col)
                sr_rows.append({"Combinaison": col, "Sharpe 30j": None if sr is None else round(sr, 4)})



        # 5. Optimisation
        if returns_df.shape[1] < 2 or returns_df.std().sum() == 0:
            st.error("⚠️ Impossible d’optimiser : pas assez de colonnes valides.")
            weights = np.ones(returns_df.shape[1]) / max(returns_df.shape[1], 1)
        else:
            try:
                weights = optimize_weights(returns_df)
            except Exception as e:
                st.error(f"⚠️ Optimisation échouée : {e}")
                weights = np.ones(returns_df.shape[1]) / returns_df.shape[1]
                st.info("ℹ️ Poids égalitaires appliqués en fallback.")

        # 6. Affichage des résultats
        if sr_rows:
            sr_df = pd.DataFrame(sr_rows).set_index("Combinaison")
        else:
            st.warning("Aucune combinaison valide pour calculer le Sharpe 30j.")

        res = pd.Series(weights, index=returns_df.columns, name="Weight")
        res_df = res.to_frame()
        merged = sr_df.join(res_df)
        merged["Weight (%)"] = (merged["Weight"] * 100).round(2)
        merged = merged.drop(columns=["Weight"])
        merged = merged.sort_values("Weight (%)", ascending=False)
        st.write("📋 Sharpe 30 jours & Poids optimaux (par combinaison symbole–stratégie)")
        st.dataframe(merged.style.format({"Sharpe 30j":"{:.4f}","Weight (%)":"{:.2f}"}))

        # 7. Perf du portefeuille
        port = returns_df.dot(weights)
        cum_eq = (1 + port).cumprod()
        mu, sigma = port.mean(), port.std()
        sharpe = mu / sigma if sigma != 0 else float("nan")

        st.metric("Sharpe Ratio (opt)", f"{sharpe:.2f}")
        st.line_chart(cum_eq)


        # 8. Détail : top 3 contributions

        w = pd.Series(weights, index=returns_df.columns, name="Weight")

        # On montre les 3 combinaisons qui, compte tenu de leur poids dans le portefeuille et de leur volatilité,
        # pèsent le plus dans le risque global (au sens “poids × volatilité”).
        # Ce n’est pas une contribution au rendement attendu — pour ça il faudrait utiliser la moyenne des rendements —
        # et ce n’est pas non plus la véritable contribution au risque prenant en compte les corrélations.
        # C’est juste un indicateur rapide, facile à lire.


        contrib = (w * returns_df.std()).sort_values(ascending=False).head(3)
        st.write("🔎 Top 3 combinaisons contributrices (approximation simple du risque)")
        contrib_df = contrib.to_frame(name="Contribution")
        st.dataframe(contrib_df)


        
        # la vraie contribution au risque (tenant compte des corrélations):
        #
        # matrice de covariance Σ :
        #
        #     MRC (Marginal Risk Contribution) : Σ @ w
        #
        #     RC (Risk Contribution) composante i : w_i * (Σ @ w)_i
        #
        #     Part de risque (%) : RC / (w.T @ Σ @ w)
        #

        cov = returns_df.cov()
        marginal = cov.dot(w)  # Σ w
        rc = w * marginal  # contributions absolues
        total_var = float(w.T.dot(marginal))
        pct_rc = (rc / total_var).sort_values(ascending=False).head(3)
        st.write("🔎 Top 3 contributions au risque (tenant compte des corrélations)")
        st.dataframe(pct_rc.to_frame(name="Part du risque (%)"))

        # associe chaque colonne à son poids
        w_series = pd.Series(weights, index=returns_df.columns)

        # décompose index en ticker et stratégie
        df_weights = (
            w_series
            .reset_index()
            .rename(columns={"index": "col", 0: "weight"})
        )
        df_weights[["Ticker", "Strategy"]] = df_weights["col"].str.split("_", n=1, expand=True)

        # pour chaque Ticker, on sélectionne la ligne où weight est maximal
        best = (
            df_weights
            .sort_values("weight", ascending=False)
            .groupby("Ticker", as_index=False)
            .first()[["Ticker", "Strategy", "weight"]]
        )
        # on renormalise pour que la somme des poids « gagnants » soit 1
        best["weight"] = (best["weight"] / best["weight"].sum())*100
        best["asof"] = pd.Timestamp.now(tz="Europe/Paris").floor("s")
        # on trie du poids le plus grand au plus petit
        best = best.sort_values("weight", ascending=False).reset_index(drop=True)

        upsert_best_strat_sqlite(best)

        st.session_state["best_strat_by_ticker"] = best
        save_best_strat_json(best)
        save_best_strat_sqlite(best)

    # Affichage du mémo
    st.subheader("📋 Mémo – Meilleure stratégie par ticker")
    """
    Affiche la meilleure stratégie pour chaque ticker avec un poids optimisé
    """
    if "best_strat_by_ticker" not in st.session_state:
        #saved = load_best_strat_json()  # ← tente de lire data/best_strat.json
        saved = load_best_strat_sqlite() # ← tente de lire data/bot_state.db
        st.session_state["best_strat_by_ticker"] = (
            saved if saved is not None else pd.DataFrame()
        )

    if "best_strat_by_ticker" in st.session_state and st.session_state["best_strat_by_ticker"] is not None:
        st.table(st.session_state["best_strat_by_ticker"] .rename(columns={
            "Ticker": "Ticker",
            "Strategy": "Stratégie retenue",
            "weight": "Poids optimisé (en %)"
        }))
        if st.button("📥 Charger dans exécution manuelle"):
            st.session_state["strategies_to_execute"] = {
                row.Ticker: (row.Strategy, row.weight) for _, row in st.session_state["best_strat_by_ticker"].iterrows()
            }
            st.success("✅ Mémo chargé – prête pour exécution.")


