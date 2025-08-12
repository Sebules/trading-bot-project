# file: execution/select_by_prediction.py
import pandas as pd
import os
import sys
import time
import requests
import streamlit as st
from datetime import datetime, timedelta
try:
    from alpaca_trade_api.rest import REST, TimeFrame
except Exception:
    REST, TimeFrame = None, None
from alpha_vantage.timeseries import TimeSeries

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY,ALPACA_PAPER_URL
from ml.train_model import load_model, get_candidate_features  # ton module d’export de modèles
from strategies.moving_average import MovingAverageStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.bollinger_bands import BBStrategy
from strategies.breakout_range_strategy import BreakoutRangeStrategy


try:
    api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_PAPER_URL)
except Exception:
  if REST is None or TradingClient is None:
        st.warning("Mode démo : les fonctionnalités Alpaca sont désactivées sur Streamlit Cloud.")
        st.info("Pour les ordres et le bot, utilisez l’environnement local / script auto_bot.py.")
        st.stop()  # ou return si c’est dans une fonction/page

BASE_URL_AV = "https://www.alphavantage.co/query"
try:
    ALPHA_API_KEY = st.secrets["AV_API_KEY"] #clé API Alpha VAntage
    #ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')
except (KeyError, AttributeError):
    st.error("🔑 Clé d'API OpenAI non trouvée. Veuillez la configurer dans `.streamlit/secrets.toml`.")
    st.stop()

def get_tradeable_symbols() -> list[str]:
    assets = api.list_assets(status="active", asset_class="us_equity")
    return [a.symbol for a in assets if a.tradable]


def get_tradeable_symbols_filtered(api: REST) -> list[str]:
    """
    Récupère une liste filtrée de symboles tradables d'Alpaca.
    """
   # On récupère tous les activs échangeables
    assets = api.list_assets(status="active", asset_class="us_equity")

    # Filtrer les actifs selon les critères de liquidité et de volatilité suggérés.
    # - Volume moyen quotidien > 500,000 actions
    # - Prix de l'action > 10$ pour éviter les penny stocks
    # - Échangeable sur les marchés majeurs
    filtered_assets = [
        a.symbol for a in assets
        if a.tradable and a.exchange in ['NYSE', 'NASDAQ']
        and a.shortable  # Un bon indicateur de liquidité
        and float(a.easy_to_borrow)  # Un autre indicateur de disponibilité
    ]

    # On peut également exclure les symboles avec des suffixes spéciaux (warrants, actions préférées, etc.)

    symbols_to_exclude = ['.', '/', '-', '+', 'W', 'P', 'R']  # Exemples de suffixes à exclure

    filtered_symbols = []
    for sym in filtered_assets:
        if not any(suffix in sym for suffix in symbols_to_exclude):
            filtered_symbols.append(sym)

    return filtered_symbols

def fetch_bulk_history(symbols: list[str], period_days: int = 365) -> dict[str, pd.DataFrame]:
    """
    Récupère l'historique des prix symbole par symbole pour une robustesse maximale.
    Retourne un dict { symbol: DataFrame }.
    """
    api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)


    end_date_test = datetime(2025, 6, 27).date().isoformat()
    start_date_test = (datetime(2025, 6, 27) - timedelta(days=10)).date().isoformat()
    try:
        test_bars = api.get_bars("AAPL", TimeFrame.Day, start=start_date_test, end=end_date_test)
        if test_bars.df.empty:
            print("WARNING: 'AAPL' data is empty for the specified date range. Check API key and dates.")
        else:
            print("SUCCESS: 'AAPL' data retrieved successfully.")
    except Exception as e:
        print(f"ERROR fetching test data for AAPL: {e}")

    hist = {}

    # Définir la plage de dates pour la requête groupée
    # On s'assure de ne demander que des données historiques complètes (jusqu'à la veille)
    # pour éviter les erreurs de souscription
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=period_days)
    start_date_iso = start_date.date().isoformat() # On ne prend que la date
    end_date_iso = end_date.date().isoformat() # On ne prend que la date

    print(f"--- Lancement de la récupération pour {len(symbols)} symboles (un par un) ---")

    # Boucle sur chaque symbole individuellement
    for i, sym in enumerate(symbols):

        try:
            # Afficher la progression tous les 20 symboles
            if (i + 1) % 100 == 0:
                print(f"   Progression : {i + 1}/{len(symbols)}")
            bars = api.get_bars(
                sym,
                TimeFrame.Day,
                start=start_date_iso,  # Utilisation de dates explicites pour la requête
                end=end_date_iso,
                adjustment="raw"  # ou "all" si vous voulez les splits/dividendes
            )
            # bars est un BarsResponse avec un DataFrame multi-indexé
            bars_df = bars.df

            if not bars_df.empty:
                # La structure du DataFrame est déjà correcte, il suffit de renommer les colonnes
                # L'index est déjà le timestamp
                bars_df = bars_df.rename(columns={
                    "open": "Open", "high": "High", "low": "Low",
                    "close": "Close", "volume": "Volume"
                })
                hist[sym] = bars_df[["Open", "High", "Low", "Close", "Volume"]]


        except Exception as e:
            # Si un symbole échoue, on affiche l'erreur et on continue
            print(f"!!!!!!!!!! Échec pour le symbole {sym}. Erreur: {e}. On continue... !!!!!!!!!!")

    print(f"--- Récupération terminée. Succès pour {len(hist)}/{len(symbols)} symboles. ---")
    return hist



def fetch_bulk_history_av(symbols: list[str], period_days: int = 365) -> dict[str, pd.DataFrame]:
    """
    Récupère l'historique des prix pour une liste de symboles via Alpha Vantage.
    Gère les limites de requêtes et les symboles non trouvés.
    """
    if ALPHA_API_KEY is None:
        return {}  # On retourne un dictionnaire vide si la clé n'est pas configurée

    hist = {}

    # Calcul de la date de début pour 1 an d'historique
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)

    print(f"Fetching data for {len(symbols)} symbols from Alpha Vantage...")

    # Alpha Vantage a une limite de 5 requêtes par minute pour la clé gratuite.
    # Nous allons attendre entre chaque requête pour respecter cette limite.
    # On peut faire des requêtes en boucle pour chaque symbole.
    for sym in symbols:
        try:
            # Alpha Vantage API pour les données historiques "daily"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": sym,
                "outputsize": "full",  # Récupère l'historique complet
                "apikey": ALPHA_API_KEY,
            }

            response = requests.get(BASE_URL_AV, params=params)
            data = response.json()

            # Gérer les erreurs de l'API Alpha Vantage
            if "Error Message" in data or "Note" in data:
                print(f"Skipping {sym}: {data.get('Error Message', data.get('Note', 'Unknown error'))}")
                continue  # Passer au symbole suivant

            # Récupérer les données de la série temporelle
            daily_data = data.get("Time Series (Daily)", {})
            if not daily_data:
                print(f"No daily data found for {sym}. Skipping.")
                continue

            # Convertir les données en DataFrame
            df = pd.DataFrame.from_dict(daily_data, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                "1. open": "Open", "2. high": "High", "3. low": "Low",
                "4. close": "Close", "5. volume": "Volume"
            })

            # Convertir les colonnes en types numériques
            df = df.astype(float)

            # Filtrer les données pour la période demandée
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            if not df.empty:
                hist[sym] = df
            else:
                print(f"Data for {sym} is outside the requested date range.")

            # Attendre pour respecter la limite de l'API (5 requêtes/min)
            time.sleep(13)  # Attendre 13 secondes (60/5 = 12, on ajoute 1 pour la sécurité)

        except Exception as e:
            print(f"Error fetching data for {sym}: {e}. Skipping.")
            continue

    return hist

def predict_top_n(model, features_list: list[str], n: int = 20, metadata: dict = None) -> list[tuple[str, float]]:
    """
    Retourne les n symboles les plus susceptibles de monter,
    avec leur probabilité prédite de classe 1 (hausse).
    """
    symbols = get_tradeable_symbols_filtered(api)

    # 1) Récupération bulk de l'historique via Alpaca (plus rapide)
    hist_dict = fetch_bulk_history(symbols, period_days=365)

    if not hist_dict:
        st.warning(
            "⚠️ Aucune donnée historique n'a pu être récupérée pour les symboles tradables. Veuillez vérifier votre connexion ou les symboles.")
        return []  # Retourne une liste vide pour éviter d'autres erreurs

    scores = []
    for sym, df in hist_dict.items():
        df_features = df.copy()

        # On récupère les paramètres depuis les métadonnées du modèle
        strat_params = metadata.get("strategy_params", {}) if metadata is not None else {}

        # On applique dynamiquement les stratégies en utilisant les paramètres sauvegardés

        if "Moving Average" in strat_params:
            params = strat_params["Moving Average"]
            df_features = MovingAverageStrategy(df_features, **params).generate_signals()

        if "RSI" in strat_params:
            params = strat_params["RSI"]
            df_features = RSIStrategy(df_features, **params).generate_signals()

        if "Bollinger Bands" in strat_params:
            params = strat_params["Bollinger Bands"]
            df_features = BBStrategy(df_features, **params).generate_signals()

        if "Breakout Range" in strat_params:
            params = strat_params["Breakout Range"]
            df_features = BreakoutRangeStrategy(df_features, **params).generate_signals()

        # on vérifie que la dernière ligne contient toutes les features

        # On reindexe notre DataFrame pour y inclure TOUTES les colonnes attendues,
        # celles qui manquent deviennent NaN, et on remplace ensuite les NaN par 0.
        df_aligned = df_features.reindex(columns=features_list)
        # On extrait la dernière ligne et on remplace les NaN par 0 (ou une autre valeur de votre choix)
        X = df_aligned.iloc[[-1]].fillna(0)

        # Si vous voulez vérifier qu'il n'y a plus de NaN :
        if X.isna().any(axis=None):
            print(f"Warning: After fillna, still missing features for {sym}, but forcing through.")

        prob = model.predict_proba(X)[0][1]  # probabilité de hausse
        scores.append((sym, prob))

    # trie et top-n

    return sorted(scores, key=lambda x: x[1], reverse=True)[:n]

