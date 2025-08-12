import pandas as pd
import numpy as np
import streamlit as st
import os, io
import sys
from pathlib import Path
import ta
from ta.volatility import BollingerBands
from itertools import product
from scipy.optimize import minimize
from datetime import datetime, timedelta, timezone
import plotly.express as px
import plotly.graph_objects as go
from decimal import Decimal


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
try:
  from alpaca.trading.client import TradingClient
except Exception:
  TradingClient = None
try:
  from alpaca.trading.requests import GetPortfolioHistoryRequest, GetOrdersRequest
except Exception:
  GetPortfolioHistoryRequest, GetOrdersRequest = None, None
try:
  from alpaca.trading.enums import QueryOrderStatus
except Exception:
  QueryOrderStatus = None
try:
  from alpaca.common.enums import Sort
except Exception:
  Sort = None

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root)

from reporting.performance_report import calculate_performance_metrics
from execution.execute_trades import get_current_position_qty, execute_signal
from execution.strategies_robot_optimizer import _normalise_returns
from execution.run_bot import load_latest_reports_by_symbol, determine_qty
from utils.chat_component import init_chat_with_emilio
from utils.portfolio import optimize_weights, generate_random_portfolios
from utils.compare_strategies import clean_name, compare_strategies
from utils.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY,ALPACA_PAPER_URL, DATA_ROOT
from utils.risk_management import (compute_dashboard_metrics, allowed_max_qty,
                                   can_place_order, reserve_order_risk, post_slack_risk_summary)
from broker.cash import compute_investable_cash
from broker.execution import build_order_plan
from broker.alpaca_executor import plot_equity_png
from broker.strategy_rules import choose_best_strategy_by_sharpe
from broker.persistence import load_best_strat_sqlite



if "plan_ready" not in st.session_state:
    st.session_state.plan_ready       = False   # le plan est-il prêt ?
    st.session_state.rebalance_plan   = []      # liste d'ordres
    st.session_state.rebalance_df_raw = None    # DataFrame brut (pas le style)
    st.session_state.rebalance_cash   = 0.0


# === CONFIG PATH ===
# On remonte maintenant de 3 niveaux pour atteindre la racine du projet
ROOT_DIR = os.path.dirname(
             os.path.dirname(                # to streamlit_app
               os.path.dirname(              # to project root
                  os.path.abspath(__file__))))

DATA_DIR = DATA_ROOT
REPORT_DIR = os.path.join(ROOT_DIR, "reporting")
RESULT_DIR = os.path.join(REPORT_DIR, "resultat")
ML_TRAIN_DIR = os.path.join(ROOT_DIR, "ml")
ML_MODELS_DIR = os.path.join(ML_TRAIN_DIR, "trained_models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(ML_TRAIN_DIR, exist_ok=True)
os.makedirs(ML_MODELS_DIR, exist_ok=True)

try:
  api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_PAPER_URL)
  client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper="Paper trading")
except Exception:
  if REST is None or TradingClient is None:
        st.warning("Mode démo : les fonctionnalités Alpaca sont désactivées sur Streamlit Cloud.")
        st.info("Pour les ordres et le bot, utilisez l’environnement local / script auto_bot.py.")
        st.stop()  # ou return si c’est dans une fonction/page

# === OPENAI CHAT -appel du chat Emilio ===
#init_chat_with_emilio()
st.title("🦙 Alpaca")
tabs = st.tabs(["🦙 En temps réel", "⚙️ Passage d’ordres","🤖 Le Bot de Trading"])

# === CREATION MODELE MACHINE LEARNING ===
with tabs[0]:
    st.header("🦙 Alpaca — En temps réel")
    if REST is None or TradingClient is None:
        st.warning("Mode démo : les fonctionnalités Alpaca sont désactivées sur Streamlit Cloud.")
        st.info("Pour les ordres et le bot, utilisez l’environnement local / script auto_bot.py.")
        st.stop()  # ou return si c’est dans une fonction/page
  
    # === PORTFOLIO EN TEMPS RÉEL ALPACA ===
    st.subheader("📦 Portefeuille en temps réel (Alpaca)")


    @st.cache_data(ttl=3600)
    def load_portfolio(_api):
        try:
            positions = _api.list_positions()
            return pd.DataFrame([p._raw for p in positions])
        except Exception as e:
            st.error(f"Erreur de chargement Alpaca : {e}")
            return pd.DataFrame()


    try:
        portfolio_df = load_portfolio(api)
        if not portfolio_df.empty:
            st.dataframe(portfolio_df[["symbol", "qty", "avg_entry_price", "market_value", "unrealized_pl"]])

        else:
            st.warning("Portefeuille vide ou aucune position active.")
    except Exception as e:
        st.warning(f"Connexion Alpaca impossible : {e}")

    account = api.get_account()
    cash = float(account.cash)
    equity = float(account.equity)
    buying_power = float(account.buying_power)
    last_equity = float(account.last_equity)
    daily_change = equity - last_equity
    daily_change_pct = (equity / last_equity) - 1.0
    st.write(f"💰 Pouvoir d'achat (paper): **${buying_power:,.2f}**")
    st.write(f"💰 Valeur totale portfolio (paper): **${equity:,.2f} - %{daily_change_pct:,.2f}**")
    st.write(f"💰 Cash disponible (paper): **${cash:,.2f}**")

    st.subheader("📊 Graphe d'Equity portfolio Alpaca")
    st.caption("Source : API Portfolio History d’Alpaca.")
    period = st.radio("Période", ["1D", "1M", "1A"], index=0, horizontal=True)
    # timeframes conseillés pour un rendu « propre »
    timeframe_by_period = {"1D": "5Min", "1M": "1D", "1A": "1D"}
    timeframe = timeframe_by_period[period]

    # Pour la crypto 24/7, Alpaca recommande:
    # intraday_reporting="continuous", pnl_reset="no_reset" (facultatif)
    # cf. docs officielles.
    hist_req = GetPortfolioHistoryRequest(
        period=period,
        timeframe=timeframe,
        extended_hours=True,           # inclut pre/post-market si < 1D
        # intraday_reporting="extended_hours",  # ou "continuous" pour crypto
        # pnl_reset="no_reset",                # utile pour crypto
    )

    try:
        hist = client.get_portfolio_history(hist_req)  # renvoie PortfolioHistory
        # -> attributs arrays: timestamp, equity, profit_loss, ...
        ts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in hist.timestamp]
        df_hist = pd.DataFrame({"equity": [float(x) for x in hist.equity]}, index=pd.to_datetime(ts))
        # Affiche en IMAGE (png)
        st.image(plot_equity_png(df_hist), caption=f"Equity – {period} ({timeframe})", use_column_width=True)
        # Option : tableau brut
        with st.expander("Voir les données (échantillon)"):
            st.dataframe(df_hist.tail(20))
        # === 🛡️ Risk Dashboard ===
        st.subheader("🛡️ Risk Dashboard")
        lookback = st.slider("Fenêtre (jours)", 20, 252, 60)
        alpha = 0.95
        k_pct = st.slider("k% de la VaR pour la taille max d'une position", 1, 50, 10)

        if st.toggle("historique daily (1M x 1D)", value=True):
            risk_hist_req = GetPortfolioHistoryRequest(
                period="1M",  # >= quelques semaines
                timeframe="1D",  # returns JOURS
                extended_hours=True
            )
            hist_risk = client.get_portfolio_history(risk_hist_req)
            ts_risk = [datetime.fromtimestamp(t, tz=timezone.utc) for t in hist_risk.timestamp]
            df_risk = pd.DataFrame({"equity": [float(x) for x in hist_risk.equity]},
                                   index=pd.to_datetime(ts_risk))

            equity_ser = df_risk["equity"].dropna().tail(lookback + 1)
        else:
            equity_ser = df_hist["equity"]
            if timeframe == "5Min":  # 1D -> 5Min
                equity_ser = equity_ser.resample("1D").last()
            equity_ser = equity_ser.dropna().tail(lookback + 1)  # +1 pour pct_change interne

        metrics = compute_dashboard_metrics(
            equity_series=equity_ser,
            equity_now=float(account.equity),
            lookback=lookback,
            alpha=alpha
        )
        st.session_state["risk_metrics"] = metrics
        st.session_state["risk_k_pct"] = k_pct

        st.metric("Volatilité annualisée (ex-ante)", f"{metrics['vol_ann'] * 100:.2f}%")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("VaR 95% (1j)", f"-${metrics['var_$']:,.0f}")
        with c2:
            st.metric("ES 95% (1j)", f"-${metrics['es_$']:,.0f}")

        if portfolio_df.empty or "symbol" not in portfolio_df.columns:
            st.info("Portefeuille vide : entre un ticker manuellement.")
            symbol_size = st.text_input("Ticker", value="AAPL")
        else:
            symbol_size = st.selectbox("Choisir le Ticker pour calculer la taille max de position",
                                       portfolio_df["symbol"])
        if symbol_size:
            try:
                last_price = float(api.get_latest_trade(symbol_size).price)
                qty_max = allowed_max_qty(last_price, metrics["var_$"], k_pct)
                st.write(f"📏 Taille max {symbol_size} : **{qty_max}** actions (k = {k_pct}%).")
            except Exception as e:
                st.warning(f"Prix introuvable pour {symbol_size}: {e}")

        # Envoi Slack (si SLACK_WEBHOOK_URL disponible)

        #Totalement optionnel. C’est juste pour :
        # pousser des alertes (ex : “VaR ↑ 50% vs hier”, “ordre bloqué par k%×VaR”),
        # loguer un snapshot de risque quand tu n’as pas l’appli ouverte.

        webhook = None
        try:
            webhook = st.secrets.get("SLACK_WEBHOOK_URL", None)
        except Exception:
            import os

            webhook = os.environ.get("SLACK_WEBHOOK_URL", None)

        if webhook and st.button("Publier ces métriques dans Slack"):
            ok = post_slack_risk_summary(webhook, metrics,
                                         symbol=symbol_size if symbol_size else None,
                                         qty_max=qty_max if "qty_max" in locals() else None,
                                         k_pct=k_pct)
            st.success("Envoyé dans Slack ✅" if ok else "Échec d'envoi Slack")
    except Exception as e:
        st.error(f"Erreur lors de la récupération de l’historique du portefeuille : {e}")

    st.subheader("📋 Liste d’ordres")
    st.caption("Liste d’ordres via TradingClient.get_orders().")
    filt_label = st.selectbox("Statut", ["Tous", "Ouverts", "Fermés"], index=0)
    status_map = {
        "Tous": QueryOrderStatus.ALL,
        "Ouverts": QueryOrderStatus.OPEN,
        "Fermés": QueryOrderStatus.CLOSED,
    }
    limit = st.slider("Nombre d’ordres", min_value=10, max_value=200, value=50, step=10)

    try:
        orders = client.get_orders(
            GetOrdersRequest(
                status=status_map[filt_label],
                limit=limit,
                direction=Sort.DESC,   # ordonner du plus récent au plus ancien
            )
        )
        rows = []
        for o in orders:
            rows.append({
                "submitted_at": getattr(o, "submitted_at", None),
                "symbol": getattr(o, "symbol", None),
                "side": getattr(o, "side", None),
                "type": getattr(o, "type", None),
                "status": getattr(o, "status", None),
                "qty": getattr(o, "qty", None),
                "notional": getattr(o, "notional", None),
                "filled_qty": getattr(o, "filled_qty", None),
                "filled_avg_price": getattr(o, "filled_avg_price", None),
                "limit_price": getattr(o, "limit_price", None),
                "stop_price": getattr(o, "stop_price", None),
                "time_in_force": getattr(o, "time_in_force", None),
            })
        df_orders = pd.DataFrame(rows)
        st.dataframe(df_orders, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors du chargement des ordres : {e}")


with tabs[1]:
    st.header("🦙 Alpaca — Passage d’ordres")
    if REST is None or TradingClient is None:
            st.warning("Mode démo : les fonctionnalités Alpaca sont désactivées sur Streamlit Cloud.")
            st.info("Pour les ordres et le bot, utilisez l’environnement local / script auto_bot.py.")
            st.stop()  # ou return si c’est dans une fonction/page
  
    st.subheader("📋 Mémo des stratégies chargées")
    if "strategies_to_execute" in st.session_state:
        st.write("Le mémo des stratégies à exécuter est bien chargé dans la session :")
        data_for_df = []
        for ticker, info_tuple in st.session_state["strategies_to_execute"].items():
            strategy_name = info_tuple[0]  # Le premier élément du tuple est la stratégie
            weight_value = info_tuple[1]  # Le deuxième élément du tuple est le poids

            data_for_df.append({
                "Ticker": ticker,
                "Stratégie": str(strategy_name),  # Conversion en string pour la sécurité
                "Poids": float(weight_value)  # Conversion en float pour la sécurité
            })

        memo_df = pd.DataFrame(data_for_df)
        st.dataframe(memo_df)
    else:
        st.info("Aucun mémo de stratégie n'a été chargé pour l'instant.")

    # --- 1) Récupération du portfolio et de l'account ---
    account = api.get_account()
    cash = float(account.cash)

    st.subheader("⚙️ Paramètres d’exécution Alpaca")

    use_pct = st.slider("% du cash à déployer", 1.0, 100.0, 25.0)


    exec_orders = st.button("🚀 Préparer l'exécution des ordres pondérés", key="prepare")


    if exec_orders:
        if use_pct <= 0:
            st.warning("Choisissez un pourcentage > 0")
            st.stop()

        try:
            memo = st.session_state["strategies_to_execute"]
        except KeyError:
            st.warning("❌ Avez-vous chargé le Mémo des stratégies à exécuter?")
            st.session_state["strategies_to_execute"] = []
            memo = st.session_state["strategies_to_execute"]

        with st.spinner("Préparation des ordres…"):
            # État actuel du compte
            positions = {p.symbol: int(p.qty) for p in api.list_positions()}


            # Plan théorique des ordres
            order_plan, recap = build_order_plan(
                api=api,
                positions=positions,
                memo=memo,  # ton dict {symbol: (strat, weight_pct)}
                cash=cash,
                pct=use_pct,
                return_recap = True
            )

            recap_df = pd.DataFrame(recap).set_index("Symbol").sort_values(by="Poids (%)",ascending=False)

            st.session_state.rebalance_plan = order_plan
            st.session_state.rebalance_df = recap_df
            st.session_state.rebalance_cash = compute_investable_cash(cash, use_pct)
            st.session_state.plan_ready = True
            st.success("Plan généré ! Vérifie le tableau ci-dessous puis confirme.")


    # Affichage
    if st.session_state.plan_ready:
        st.subheader("📋 Plan de rééquilibrage (pré-visualisation)")

        def highlight(row):
            return ["background-color:#ffdddd" if row["Δ Qty"] else "" for _ in row]

        recap_df_styled = (
            st.session_state.rebalance_df
            .style
            .apply(highlight, axis=1)
            .format({"Poids (%)": "{:.3f}","Prix": "{:.2f}", "Valeur cible $": "{:,.2f}"})
        )
        st.dataframe(recap_df_styled, use_container_width=True)

        # Contrôle de budget temps-réel
        remaining_cash = compute_investable_cash(cash, use_pct)

        # Boucle d’envoi *effectif* des ordres
        if st.button("✅ Confirmer et envoyer les ordres", key="confirm"):
            remaining_cash = st.session_state.rebalance_cash

            for od in st.session_state.rebalance_plan:
                # -- od == {"symbol": "AAPL", "side": "buy"/"sell", "qty": 12, "est_cost": 2 384.16}
                cost = od["est_cost"]
                symbol = od["symbol"]

                # a) On vérifie qu’on reste dans le budget
                if cost > remaining_cash and od["side"] == "buy":
                    st.warning(f"⛔ Budget insuffisant pour {symbol} : "
                               f"il reste seulement {remaining_cash:,.2f}$")
                    continue  # on saute l'ordre, on ne stoppe pas toute la boucle

                # b) On récupère le dernier prix pour le limit order
                price = float(api.get_latest_trade(symbol).price)

                # c) On soumet l’ordre
                api.submit_order(
                    symbol=symbol,
                    qty=od["qty"],
                    side=od["side"],
                    type="limit", #passer un ordre à limite
                    limit_price=round(price,2),
                    time_in_force="day",
                )

                # d) Suivi visuel + update du budget
                st.success(f"✅ {od['side'].upper()} {od['qty']} × {symbol} envoyé "
                           f"(≈ {cost:,.2f}$)")
                if od["side"] == "buy":
                    remaining_cash -= cost

            st.session_state.plan_ready = False
            st.session_state.rebalance_plan = []
            st.session_state.rebalance_df_raw = None
            st.session_state.rebalance_cash = 0.0
            st.success("Tous les ordres ont été soumis ✅")


with tabs[2]:
    # === SECTION D'EXÉCUTION DU BOT AUTOMATIQUE ===
    st.subheader("🤖 Contrôle du Bot de Trading")
    if REST is None or TradingClient is None:
                st.warning("Mode démo : les fonctionnalités Alpaca sont désactivées sur Streamlit Cloud.")
                st.info("Pour les ordres et le bot, utilisez l’environnement local / script auto_bot.py.")
                st.stop()  # ou return si c’est dans une fonction/page
  
    # Activation manuelle depuis l'interface
    activer_bot = st.toggle("🟢 Activer le bot sur le dernier rapport", value=False)

    if activer_bot:
        st.info("⚙️ Le bot est prêt à s'exécuter.")
        # ✅ Nouveaux réglages d’allocation
        use_memo_weights = st.toggle("Utiliser le mémo d’optimisation pour le sizing", value=True)
        follow_memo_strategy = st.toggle("Suivre la stratégie du mémo (sinon: meilleur Sharpe 30j)", value=True)
        use_pct_bot = st.slider("% du cash à déployer (bot)", 1.0, 100.0, 25.0)

        if st.button("🚀 Lancer maintenant le bot sur le dernier rapport"):
            try:
                reports_by_symbol = load_latest_reports_by_symbol()
                st.write("📋 reports_by_symbol:", reports_by_symbol)

                # ——— Récupère le mémo (session → sinon SQLite)
                memo = st.session_state.get("strategies_to_execute")
                if not memo or not isinstance(memo, dict) or len(memo) == 0:
                    try:
                        best = load_best_strat_sqlite()
                        if best is not None and not best.empty:
                            memo = {row.Ticker: (row.Strategy, float(row.weight)) for _, row in best.iterrows()}
                        else:
                            memo = {}
                    except Exception:
                        memo = {}

                # ——— Capital déployable pour dimensionner les poids
                account = api.get_account()
                deployable_cash = compute_investable_cash(float(account.cash), use_pct_bot)


                for symbol, rapport_path in reports_by_symbol.items():
                    # 👇 Vérifier ici si Alpaca connaît le symbole, sinon on l'affiche en warning et on skip
                    try:
                        asset = api.get_asset(symbol)
                        if not asset.tradable:
                            st.warning(f"⚠️ {symbol} n’est pas tradable sur Alpaca – passage au symbole suivant.")
                            continue
                    except Exception as e:
                        st.warning(f"⚠️ Symbole invalide ou inaccessible : {symbol} ({e}) – ignoré.")
                        continue
                    st.markdown(f"### ⚙️ Traitement pour **{symbol}**")

                    df_temp = pd.read_csv(rapport_path, index_col=0, parse_dates=True)
                    df_temp.index = pd.to_datetime(df_temp.index, utc=True).tz_convert(None)
                    last_price = df_temp["Close"].iloc[-1]


                    # 🔎 Sélection de la stratégie
                    memo_strat = memo.get(symbol, (None, None))[0] if use_memo_weights else None
                    if follow_memo_strategy and memo_strat:
                        # On suit la stratégie choisie par l’optimisation (si elle existe dans le rapport)
                        sig_col_candidate = f"{memo_strat}_Signal"
                        if sig_col_candidate in df_temp.columns:
                            best_name, best_sig_col, best_ret_col, best_sr = memo_strat, sig_col_candidate, f"{memo_strat}_Returns", None
                        else:
                            # fallback: meilleur Sharpe si la colonne du mémo n'existe pas
                            best_name, best_sig_col, best_ret_col, best_sr = choose_best_strategy_by_sharpe(df_temp)
                    else:
                        best_name, best_sig_col, best_ret_col, best_sr = choose_best_strategy_by_sharpe(df_temp)

                    if best_sig_col is None:
                        st.info("ℹ️ Aucune paire <Stratégie>_Returns/_Signal détectée "
                                "— on ne peut pas appliquer la règle Sharpe 30j.")
                    else:
                        if (best_sr is None) and not follow_memo_strategy:
                            st.info(f"ℹ️ Pas assez de données pour calculer le Sharpe 30j de {symbol}"
                                    f" — pas de remplacement automatique.")
                        elif (best_sr is not None) and (best_sr < 0):
                            st.warning(f"⛔ {symbol} : Sharpe 30j négatif pour toutes les stratégies"
                                       f" (meilleur = {best_sr:.2f}). Aucun ordre ne sera proposé.")
                            # on passe au symbole suivant (pas d'UI quantité/bouton pour ce symbole)
                            continue
                        else:
                            st.success(f"✅ {symbol} : stratégie sélectionnée « {best_name} » "
                                       f"{'' if best_sr is None else f' (Sharpe 30j = {best_sr:.2f})'}.")

                    # ——— Quantité par défaut : poids du mémo → sinon sizing classique
                    default_qty, cash = determine_qty(symbol, mode="paper", last_price=last_price)
                    if use_memo_weights and symbol in memo and memo[symbol][1] is not None:
                        weight_pct = float(memo[symbol][1])  # déjà en %
                        target_notional = (weight_pct / 100.0) * deployable_cash
                        qty_from_weight = int(max(0, target_notional // float(last_price)))
                        if qty_from_weight > 0:
                            default_qty = qty_from_weight

                        # Affichage du cash disponible
                    st.write(f"💰 Cash disponible (paper): **${cash:,.2f}**")

                    state_key = f"qty_{symbol}"
                    default_value = max(0, int(default_qty))

                    # purge/sanitarise la valeur mémorisée si elle existe
                    if state_key in st.session_state:
                        if not isinstance(st.session_state[state_key], int) or st.session_state[state_key] < 0:
                            st.session_state[state_key] = default_value

                    raw_max = int(cash // last_price) if last_price > 0 else 1
                    max_val = max(0, raw_max)  # empêche un max_value négatif (arrive quand cash<0)

                    # >> input pour ajuster la quantité (pré-remplie via poids si dispo)
                    qty = st.number_input(
                        f"Quantité pour {symbol}",
                        min_value=0,
                        max_value=max_val,
                        value=st.session_state.get(state_key, default_value),
                        step=1,
                        key=state_key  # clé unique
                    )

                    # >> bouton dédié par symbole
                    if st.button(f"🚀 Exécuter ordre pour {symbol}", key=f"btn_{symbol}"):
                        # on relit le DataFrame pour récupérer le signal
                        df = pd.read_csv(rapport_path, index_col=0, parse_dates=True)
                        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)


                        # 🔁 Re-sélection/contrôle au clic (robuste si le fichier a changé)
                        if follow_memo_strategy and memo_strat:
                            cand = f"{memo_strat}_Signal"
                            if cand in df.columns:
                                best_name, best_sig_col, best_ret_col, best_sr = memo_strat, cand, f"{memo_strat}_Returns", None
                            else:
                                best_name, best_sig_col, best_ret_col, best_sr = choose_best_strategy_by_sharpe(df)
                        else:
                            best_name, best_sig_col, best_ret_col, best_sr = choose_best_strategy_by_sharpe(df)

                        if best_sig_col is None:
                            # fallback minimaliste si aucune colonne détectée
                            signal_cols = [c for c in df.columns if c.endswith("_Signal")] or ["Signal"]
                            signal_col = signal_cols[0]
                            last_signal = int(df[signal_col].dropna().iloc[-1])
                        else:
                            if (best_sr is not None) and (best_sr < 0):
                                st.warning(
                                    f"⛔ {symbol} : Sharpe 30j négatif (meilleur = {best_sr:.2f}). Pas d’exécution.")
                                continue
                            signal_col = best_sig_col
                            last_signal = int(df[signal_col].dropna().iloc[-1])

                        # >> vérification du cash avant ordre
                        cost = qty * last_price

                        risk_metrics = st.session_state.get("risk_metrics")
                        risk_k = st.session_state.get("risk_k_pct", 10)

                        if not risk_metrics:
                            st.warning(
                                "Initialise le Risk Dashboard (onglet En temps réel) pour activer les garde-fous.")
                        else:
                            var_dollars = float(risk_metrics.get("var_$") or 0.0)
                            budget = (risk_k / 100.0) * abs(var_dollars)  # budget = k% × VaR($)
                            if not can_place_order(cost, k_pct=risk_k, portfolio_var_dollars=risk_metrics["var_$"]):
                                st.error(f"⛔ Ordre rejeté : coût ≈ ${cost:,.0f} "
                                         f"> budget {risk_k:.0f}% × VaR (≈ ${budget:,.0f} ; VaR=${var_dollars:,.0f})."
                                         )
                                continue
                            reserve_order_risk(cost)  # on ‘réserve’ le risque du jour
                        if cost > cash:
                            st.error(f"💥 Coût de l’ordre (${cost:,.2f}) supérieur au cash dispo (${cash:,.2f}).")
                        else:
                            try:
                                if last_signal == 1:
                                    api.submit_order(symbol=symbol, qty=qty, side="buy",
                                                     type="market", time_in_force="gtc")
                                elif last_signal == -1:
                                    api.submit_order(symbol=symbol, qty=qty, side="sell",
                                                     type="market", time_in_force="gtc")
                                else:
                                    st.info("ℹ️ Signal = 0 → pas d’action.")
                                    continue

                                st.success(f"✅ {symbol} ➤ Signal {last_signal} exécuté ({qty} actions)")
                            except Exception as e:
                                st.error(f"❌ Erreur exécution pour {symbol}: {e}")
            except Exception as e:
                st.error("❌ Impossible d'importer le module `run_bot.py`.")
                st.exception(e)
        # else:

    else:
        st.warning("⛔ Le bot est désactivé.")
