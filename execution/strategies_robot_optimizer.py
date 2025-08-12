from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Type

import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from reporting.performance_report import calculate_performance_metrics

def _load_dataframe(path: Path | str) -> pd.DataFrame | None:
    """Read *path* as a CSV and return a cleaned DataFrame or *None* if empty."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    df = df[~df.index.duplicated(keep="first")]
    return None if df.empty else df

def _normalise_returns(df_sig: pd.DataFrame) -> pd.DataFrame:
    """S'assure que ``Strategy_Returns`` column existe dans *df_sig* et retourne *df_sig*."""
    # if "Strategy_Returns" in df_sig.columns:
    #     return df_sig
    #
    # if "Equity_Curve" in df_sig.columns:
    #     df_sig["Strategy_Returns"] = df_sig["Equity_Curve"].pct_change().fillna(0)
    # elif "Returns" in df_sig.columns:
    #     df_sig["Strategy_Returns"] = df_sig["Returns"].fillna(0)
    # elif "Strategy" in df_sig.columns:
    #     df_sig["Strategy_Returns"] = df_sig["Strategy"].fillna(0)
    # else:
    mkt = df_sig["Close"].pct_change().fillna(0)
    df_sig["Strategy_Returns"] = df_sig["Signal"].shift(1).fillna(0) * mkt

    return df_sig

def run_strategy_optimizer(
    *,
    selected_tickers: List[str],
    selected_strategies: List[str],
    param_grid: Mapping[str, Mapping[str, range]],
    data_files_map: Mapping[str, str],
    strategy_choices: Mapping[str, Type],
    data_dir: str | Path,
) -> None:
    if not selected_tickers:
        st.warning("⚠️ Veuillez sélectionner au moins un symbole de données.")
    else:
        session_top3 = {}
        all_symbols_results = {}

    for ticker in selected_tickers:
        st.subheader(f"⚙️ Lancement de l'optimisation pour **{ticker}**")

        df_path = Path(data_dir) / data_files_map[ticker]
        df = _load_dataframe(df_path)
        if df is None:
            st.warning(f"⚠️ Le fichier de données pour {ticker} est vide. Ignoré.")
            continue

        all_signals_dfs = {}
        results = []

        for strat_name in selected_strategies:
            if strat_name not in strategy_choices or strat_name not in param_grid:
                continue  # aucun paramètre défini

            strat_cls = strategy_choices[strat_name]
            keys, vals = zip(*param_grid[strat_name].items())

            for combo in product(*vals):
                params = dict(zip(keys, combo))
                strat = strat_cls(df.copy(), **params)
                df_sig = strat.generate_signals()

                df_sig = _normalise_returns(df_sig)

                # Metriques de perf
                perf = calculate_performance_metrics(df_sig, strategy_col="Strategy_Returns")
                perf["Strategy"] = strat_name
                safe_name = strat_name.replace(" ", "_")
                for k, v in params.items():
                    perf[f"{safe_name}_{k}"] = v

                results.append(perf)
                all_signals_dfs[f"{strat_name}_{combo}"] = df_sig  # facultatif

        if not results:
                    st.warning(f"⚠️ Aucune stratégie n'a pu être générée pour **{ticker}**.")
                    continue

        df_res = pd.DataFrame(results)
        top_per_strategy = (
            df_res.sort_values("Sharpe Ratio", ascending=False)
            .drop_duplicates(subset=["Strategy"], keep="first")
            .set_index("Strategy")
        )

        st.subheader(f"🏆 Meilleures configs par stratégie pour {ticker}")
        st.dataframe(top_per_strategy)

        st.markdown("**🔎 Meilleure combinaison par stratégie :**")
        for strategy_name, row in top_per_strategy.iterrows():
            sharpe = row["Sharpe Ratio"]
            safe_name = strategy_name.replace(" ", "_")
            params = {k: row[f"{safe_name}_{k}"] for k in param_grid[strategy_name].keys()}
            st.markdown(f"- **{strategy_name}** (Sharpe : {sharpe:.2f}) — paramètres : {params}")

        session_top3[ticker] = top_per_strategy
        all_symbols_results[ticker] = df_res

    if session_top3:
        st.session_state.setdefault("top3_configs", {}).update(session_top3)
    st.session_state["all_strategies_results"] = all_symbols_results

    st.success("✅ Optimisation des stratégies terminée pour tous les symboles sélectionnés.")