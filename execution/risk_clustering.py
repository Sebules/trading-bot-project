import pandas as pd
import time
import logging
import os
import sys
try:
    from alpaca_trade_api.rest import REST, TimeFrame
except Exception:
    REST, TimeFrame = None, None
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go


logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY,ALPACA_PAPER_URL



def get_tradeable_symbols(api: REST) -> list[str]:
    """
    Récupère depuis Alpaca la liste des symboles actifs et tradables.
    """
    assets = api.list_assets(status="active")

    # on ne veut que les symboles (strings)
    return [a.symbol for a in assets if a.tradable]

def fetch_daily_returns(symbols: list[str], period_days: int = 252) -> pd.DataFrame:
    """
    Récupère en une seule requête (en batch) les rendements journaliers pour une liste de symboles.
    """
    api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_PAPER_URL)
    # On s'assure de ne demander que des données historiques complètes (jusqu'à la veille)
    # pour éviter les erreurs de souscription
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=period_days + 1)

    symbols = [s for s in symbols if "/" not in s]
    if not symbols:
        raise ValueError("Aucun symbole compatible avec Alpaca dans la liste fournie")

    # get_bars accepte une liste de symboles pour un appel groupé
    # L’API Alpaca impose ≈200 symboles max. par requête ; on découpe donc en batches.
    MAX_BATCH= 200
    chunks = [symbols[i:i + MAX_BATCH] for i in range(0, len(symbols), MAX_BATCH)]
    frames = []
    for chunk in chunks:
        try:
            df_chunk = api.get_bars(
                chunk,
                TimeFrame.Day,
                start=start_date.date().isoformat(),
                end=end_date.date().isoformat(),
                adjustment="raw"
            ).df
            if df_chunk.empty:
                continue
            df_chunk = df_chunk.reset_index()
            # Si l'API ne renvoie qu'un seul symbole, la colonne 'symbol' n'existe pas :
            if 'symbol' not in df_chunk.columns:
                df_chunk['symbol'] = chunk[0]
            frames.append(df_chunk)
        except Exception as err:
            logger.warning("Lot ignoré (%s) : %s", ','.join(chunk), err)
            continue

        time.sleep(0.2)  # petite pause pour la rate‑limit
    if not frames:
        raise RuntimeError("Aucune donnée récupérée pour les symboles donnés")
    bars = pd.concat(frames, ignore_index=True)

    # On ne garde que le prix de clôture et on pivote le DataFrame
    close_prices = bars.pivot(index='timestamp', columns='symbol', values='close')
    return close_prices.pct_change().dropna(how="all")

def cluster_and_select(symbols: list[str], n_clusters: int = 5, top_k: int = 1) -> list[str]:
    """
    Récupère pour chaque symbole sa série de prix historiques,
    calcule moyenne et volatilité, réalise un KMeans et
    sélectionne pour chaque cluster les top_k moins volatils.
    """
    # 1) Rassembler les returns journaliers via un appel groupé à Alpaca

    df_ret = fetch_daily_returns(symbols).dropna(axis=1, how="any")
    if df_ret.shape[1] == 0:
        raise RuntimeError("Aucun symbole avec historique complet après suppression des NaN")

        # 2) Calcul des stats
    stats = pd.DataFrame({
        "mean": df_ret.mean(), # rendements moyens par période
        "vol":  df_ret.std() # volatilité par période (écart-type)
    }).dropna()

    # 3) Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(stats)
    stats["cluster"] = kmeans.labels_

    # 4) Sélection
    selected = []
    for c in range(n_clusters):
        sub = stats[stats["cluster"] == c]
        tops = sub.nsmallest(top_k, "vol").index.tolist()
        selected.extend(tops)
    clusters = {c: stats.loc[stats["cluster"] == c].index.tolist() for c in range(n_clusters)}
    return selected, clusters,stats

def prepare_cluster_stats_df(stats: pd.DataFrame, selected) -> pd.DataFrame:
    # ─── Construction du DataFrame pour Plotly ───
    stats = stats.copy()
    stats.index.name = "ticker"
    stats = stats.reset_index()  # l’index (tickers) devient une colonne "ticker"
    stats["is_best"] = stats["ticker"].isin(selected)
    return stats

def build_cluster_scatter(stats: pd.DataFrame, k_clusters: int, search, *,render_mode: str = "svg",): # <= forcer le SVG pour que le nuage de point soit en-dessous.
    # Dates utilisées par le clustering (1 an)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=252 + 1)

    # ─── Nuage de points interactif ───
    fig = px.scatter(stats, x="mean", y="vol",
                     color="cluster",
                     hover_name="ticker",
                     opacity=0.7,
                     title=(f"Clusters rendement/volatilité  —  "
                            f"{start_date.date()} → {end_date.date()}   |   n_clusters = {k_clusters}"),
                     render_mode=render_mode
                     )
    # Trace des meilleurs tickers (★ rouges)
    best_df = stats[stats["is_best"]]
    fig.add_trace(
        go.Scatter(
            x=best_df["mean"],
            y=best_df["vol"],
            mode="markers",
            marker=dict(symbol="star", size=14, color="red"),
            hovertext=best_df["ticker"],
            name="Best",
            showlegend=True
        )
    )

    # ─── Recherche d’un ticker ───
    if search:
        search = search.upper().strip()
        if search in stats_df["ticker"].values:
            row = stats.loc[stats["ticker"] == search].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[row["mean"]],
                    y=[row["vol"]],
                    mode="markers+text",
                    marker=dict(size=18, symbol="diamond", color="black"),
                    text=[search],
                    textposition="top center",
                    hovertext=[row[["mean", "vol"]]],
                    name="ticker recherché",
                    showlegend=True
                )
        )

        # Zoom autour du point recherché
        x_pad = (stats["mean"].max() - stats["mean"].min()) * 0.05
        y_pad = (stats["vol"].max() - stats["vol"].min()) * 0.05
        fig.update_xaxes(range=[row["mean"] - x_pad, row["mean"] + x_pad])
        fig.update_yaxes(range=[row["vol"] - y_pad, row["vol"] + y_pad])


    return fig
