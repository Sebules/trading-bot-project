
import os
import argparse
import pandas as pd
from datetime import datetime
import sys
import re
from alpaca_trade_api.rest import REST


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from reporting.performance_report import generate_verdict
from utils.performance import compute_performance
from utils.settings import (ALPACA_API_KEY, ALPACA_SECRET_KEY,ALPACA_PAPER_URL,
                             DATA_ROOT, REPORT_ROOT, RESULT_ROOT)
from utils.risk_management import _load_daily_risk_used, _save_daily_risk_used


# ==== ALPACA CONFIG ====

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_PAPER_URL)

# ==== DIRECTORIES ====
DATA_DIR = DATA_ROOT
REPORT_DIR = REPORT_ROOT

# ==== UTILS ====
def extract_symbol_from_filename(filename):
    #match = re.match(r"rapport_([A-Z]+)_\d{8}_\d{6}\.csv", filename)
    match = re.match(r"rapport_([^_]+)_", filename)
    return match.group(1) if match else None

def extract_symbol_from_datafilename(filename):
    match = re.match(r"([^_]+)_", filename)
    return match.group(1) if match else None

def load_latest_reports_by_symbol():
    """
    Parcourt REPORT_DIR et, pour chaque symbole, conserve
    uniquement le CSV le plus récent commençant par 'rapport_<SYMBOLE>_'.
    Renvoie un dict { symbole: chemin_complet_vers_le_fichier }.
    """
    symbol_map = {}
    for f in os.listdir(REPORT_DIR):
        if not f.startswith("rapport_") or not f.endswith(".csv"):
            continue

        symbol = extract_symbol_from_filename(f)
        if not symbol:
            continue

        full_path = os.path.join(REPORT_DIR, f)
        try:
            mtime = os.path.getmtime(full_path)
        except OSError:
            # si on ne peut pas lire le fichier, on skip
            continue

        # Si on n’a pas encore enregistré de rapport pour ce symbole
        # ou si celui-ci est plus récent, on le remplace
        prev = symbol_map.get(symbol)
        if prev is None or mtime > os.path.getmtime(prev):
            symbol_map[symbol] = full_path

    return symbol_map  # { "AAPL": "/.../rapport_AAPL_2025..." }

def load_latest_data_by_symbol():
    """
    Parcourt DATA_DIR et, pour chaque symbole, conserve
    uniquement le CSV le plus récent commençant par '<SYMBOLE>_'.
    Renvoie un dict { symbole: chemin_complet_vers_le_fichier }.
    """
    symbol_map = {}
    for f in os.listdir(DATA_DIR):
        if not f.endswith(".csv"):
            continue

        symbol = extract_symbol_from_datafilename(f)
        if not symbol:
            continue

        full_path = os.path.join(DATA_DIR, f)
        try:
            mtime = os.path.getmtime(full_path)
        except OSError:
            # si on ne peut pas lire le fichier, on skip
            continue

        # Si on n’a pas encore enregistré de data pour ce symbole
        # ou si celui-ci est plus récent, on le remplace
        prev = symbol_map.get(symbol)
        if prev is None or mtime > os.path.getmtime(prev):
            symbol_map[symbol] = full_path

    return symbol_map  # { "AAPL": ." }


def determine_qty(symbol, mode,last_price, capital=10000, pct=0.02):
    """
    Exemple simple : alloue pct (%) du capital actuel à chaque ordre.
    """
    if mode == "paper":
        acct = api.get_account()
        cash = float(acct.cash)

    else:
        cash = capital
    default_qty = max(1, int((cash * pct) / last_price))  # arrondi à l’entier

    # 👇 Vérifier que le symbole existe et est tradable
    try:
        asset = api.get_asset(symbol)
        if not asset.tradable:
            print(f" {symbol} n’est pas tradable")
            price = 0
        else:
            price = float(api.get_latest_trade(symbol).price)
    except Exception as e:
        print(f"Symbole invalide ou inaccessible : {symbol} ({e})")
        price = 0

    # Prix unitaire
    # price = float(api.get_latest_trade(symbol).price)

    # 👇 Garde-fou : max 1% du capital par trade
    max_risk_per_trade = 0.01 * cash
    if price > 0:
        max_qty_by_risk = int(max_risk_per_trade // price)
    else:
        max_qty_by_risk = 0

    qty = min(default_qty, max_qty_by_risk)

    # 👇 Garde-fou journalier (à stocker en session ou fichier)
    daily_risk_used = _load_daily_risk_used()  # implémenter la persistance
    if (daily_risk_used + qty * price) > 0.03 * cash:
        qty = 0  # risque journalier dépassé
    else:
        # 👇 Mémorise ce nouveau risque
        _save_daily_risk_used(qty * price)
    return qty,cash

def execute_best_signals(symbol, rapport_path, mode="paper", log=print):

    log(f"\n📄 Rapport détecté : {os.path.basename(rapport_path)}")
    log(f"🔍 Symbole : {symbol}")

    df = pd.read_csv(rapport_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

    equity_cols = [col for col in df.columns if col.endswith("_Equity")]
    return_cols = [col.replace("_Equity", "_Returns") for col in equity_cols]

    # 1) Comparer les stratégies
    perf_results = {}
    for equity, ret in zip(equity_cols, return_cols):
        if ret in df.columns:
            strat_name = equity.replace("_Equity", "")
            perf = compute_performance(df, strategy_col=ret)
            verdict, details = generate_verdict(perf)
            perf["Verdict"] = verdict
            perf["Motifs"] = ", ".join(details)
            perf_results[strat_name] = perf

    df_perf = pd.DataFrame(perf_results).T.sort_values("Sharpe Ratio", ascending=False)

    # 2) Exécution
    for strat_name in df_perf.index:
        signal_col = f"{strat_name}_Signal"
        if signal_col not in df.columns:
            print(f"⚠️ Colonne de signal manquante pour {strat_name}")
            continue

        signal = int(df[signal_col].dropna().iloc[-1])
        qty = determine_qty(symbol, mode)
        try:
            if signal == 1:
                api.submit_order(symbol=symbol, qty=qty, side="buy", type="market", time_in_force="gtc")
            elif signal == -1:
                api.submit_order(symbol=symbol, qty=qty, side="sell", type="market", time_in_force="gtc")

            log(f"✅ {symbol} ➤ Signal {signal} exécuté via {strat_name}, quantité {qty}")
        except Exception as e:
            log(f"❌ Erreur pour {strat_name} sur {symbol} ➤ {e}")

# ==== MAIN ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trading bot daily")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    args = parser.parse_args()

    print("🚀 Bot lancé à", datetime.now())
    reports = load_latest_reports_by_symbol()
    if not reports:
        print("❌ Aucun rapport détecté.")
    else:
        for symbol, path in reports.items():
            try:
                execute_best_signals(symbol, path,mode=args.mode, log=print)
            except Exception as e:
                print(f"❌ Erreur lors de l'exécution pour {symbol} ➤ {e}")

    print("✅ Terminé à", datetime.now())