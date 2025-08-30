# scripts/auto_bot.py
"""
Bot automatique:
- MODE 'rebalance' : rééquilibrage pondéré via le mémo (comme l'onglet "Passage d’ordres")
- MODE 'sharpe'    : sélection par Sharpe 30 j (sans poids mémo)
Garde-fou: k% × VaR portefeuille (1j)
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import logging
import traceback

# --- .env facultatif ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- chemins/projet ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- imports projet (sans Streamlit) ---
from alpaca_trade_api.rest import REST
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest

from utils.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER_URL
from broker.cash import compute_investable_cash
from broker.execution import build_order_plan
from broker.persistence import load_best_strat_sqlite
from broker.strategy_rules import choose_best_strategy_by_sharpe
from broker.metrics import sr30_from_report
from execution.run_bot import load_latest_reports_by_symbol, determine_qty
from utils.risk_management import compute_dashboard_metrics, can_place_order, reserve_order_risk

# --- Logging fichier dédié (en plus des prints) ---
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "auto_bot_py.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
logging.info("=== auto_bot boot ===")

def env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

# --- config via env ---
MODE = os.getenv("AUTO_BOT_MODE", "rebalance").lower()   # 'rebalance' ou 'sharpe'
DEPLOY_PCT = float(os.getenv("AUTO_BOT_DEPLOY_PCT", "25"))  # % cash à déployer en mode rebalance
K_PCT = float(os.getenv("RISK_K_PCT", "1000"))                # k% VaR
SLEEP_SEC = int(os.getenv("AUTO_BOT_SLEEP", "60"))
ONLY_MARKET = env_bool("AUTO_BOT_ONLY_MARKET", True)
LOOP = env_bool("AUTO_BOT_LOOP", True)
PAPER = env_bool("ALPACA_PAPER", True)
BLOCK_SR = env_bool("AUTO_BOT_BLOCK_SR", True)

# --- clients Alpaca ---
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_PAPER_URL)
client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper="Paper trading" if PAPER else False)

def now_utc():
    return datetime.now(timezone.utc)

def market_open_now_utc():
    """Fenêtre US 9:30–16:00 ET ≈ 13:30–20:00 UTC (approx)."""
    t = now_utc()
    if t.weekday() >= 5:
        return False
    mins = t.hour * 60 + t.minute
    return (13 * 60 + 30) <= mins <= (20 * 60)

def print_config():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    msg = (f"[auto_bot] now={now} mode={MODE} loop={LOOP} only_mkt={ONLY_MARKET} "
           f"sleep={SLEEP_SEC}s deploy={DEPLOY_PCT}% k%VaR={K_PCT} paper={PAPER}")
    print(msg, flush=True);
    logging.info(msg)

def get_equity_daily_series(lookback_days: int = 60) -> pd.Series:
    """Equity en 1D pour une VaR/ES 1 jour cohérente."""
    hist_req = GetPortfolioHistoryRequest(period="1M", timeframe="1D", extended_hours=True)
    hist = client.get_portfolio_history(hist_req)
    ts = pd.to_datetime([datetime.fromtimestamp(t, tz=timezone.utc) for t in hist.timestamp])
    df = pd.DataFrame({"equity": [float(x) for x in hist.equity]}, index=ts)
    return df["equity"].dropna().tail(lookback_days + 1)

def risk_metrics_var_dollars() -> float:
    """VaR portefeuille en dollars (1j, 95%)."""
    account = api.get_account()
    equity_ser = get_equity_daily_series(lookback_days=60)
    metrics = compute_dashboard_metrics(
        equity_series=equity_ser,  # on passe l'equity (pct_change interne)
        equity_now=float(account.equity),
        lookback=60,
        alpha=0.95
    )
    return float(metrics.get("var_$") or 0.0)

def run_rebalance_once():
    """Rééquilibrage pondéré via le mémo (comme 'Passage d’ordres')."""
    logging.info("run_rebalance_once() start")
    account = api.get_account()
    cash_total = float(account.cash)
    deployable_cash = compute_investable_cash(cash_total, DEPLOY_PCT)

    memo_df = load_best_strat_sqlite()  # colonnes: Ticker, Strategy, weight
    if memo_df is None or memo_df.empty:
        print("[auto_bot] rebalance: memo introuvable/vide -> stop.", flush=True)
        logging.warning("rebalance: memo empty")
        return

    memo = {row.Ticker: (row.Strategy, float(row.weight)) for _, row in memo_df.iterrows()}
    positions = {p.symbol: int(p.qty) for p in api.list_positions()}

    order_plan, recap = build_order_plan(
        api=api,
        positions=positions,
        memo=memo,           # {symbol: (strategy, weight_pct)}
        cash=cash_total,
        pct=DEPLOY_PCT,
        return_recap=True
    )
    print(f"[auto_bot] rebalance: {len(order_plan)} orders proposed.", flush=True)
    logging.info("rebalance: %d orders proposed", len(order_plan))

    var_dollars = risk_metrics_var_dollars()  # 0.0 si non calculable
    # Rapports pour le Sharpe 30 j (stratégie du mémo)
    reports_by_symbol = load_latest_reports_by_symbol()
    remaining_cash = deployable_cash

    for od in order_plan:
        symbol = od["symbol"]
        side = od["side"]
        qty = int(od["qty"])
        cost = float(od.get("est_cost", 0.0))
        if qty <= 0:
            continue

        if side == "buy" and cost > remaining_cash:
            print(f"[auto_bot] {symbol}: budget cash insuffisant (reste {remaining_cash:.2f}$).", flush=True)
            logging.info("%s: cash budget insufficient (remain=%.2f)", symbol, remaining_cash)
            continue

        # Bloque les achats si Sharpe 30 j < 0
        if side == "buy" and BLOCK_SR:
            sr = None
            if (symbol in reports_by_symbol) and (symbol in memo):
                strat_name = str(memo[symbol][0])
                sr = sr30_from_report(reports_by_symbol[symbol], strat_name)
            if (sr is not None) and (sr < 0):
                print(f"[auto_bot] {symbol}: BLOCKED Sharpe30j {sr:.2f} < 0.", flush=True)
                logging.info("%s: blocked by Sharpe30j %.2f", symbol, sr)
                continue

        if var_dollars > 0 and not can_place_order(cost, k_pct=K_PCT, portfolio_var_dollars=var_dollars):
            budget = (K_PCT / 100.0) * abs(var_dollars)
            print(f"[auto_bot] {symbol}: BLOCKED k%×VaR (coût {cost:.0f}$ > budget {budget:.0f}$).", flush=True)
            logging.info("%s: blocked by k%%xVaR cost=%.0f budget=%.0f", symbol, cost, budget)
            continue

        try:
            last_price = float(api.get_latest_trade(symbol).price)
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="limit",
                limit_price=round(last_price, 2),
                time_in_force="day",
            )
            if var_dollars > 0 and side == "buy":
                reserve_order_risk(cost)
            if side == "buy":
                remaining_cash -= cost
            print(f"[auto_bot] {symbol}: {side.upper()} {qty} @ ~{last_price:.2f} OK (cost≈{cost:.0f}$)", flush=True)
            logging.info("%s: %s %d @ ~%.2f OK cost≈%.0f", symbol, side.upper(), qty, last_price, cost)
        except Exception as e:
            print(f"[auto_bot] {symbol}: erreur envoi: {e}", flush=True)
            logging.exception("%s: send error: %s", symbol, e)

def run_sharpe_once():
    """Mode Sharpe 30 j (sans utiliser les poids du mémo)."""
    logging.info("run_sharpe_once() start")
    var_dollars = risk_metrics_var_dollars()
    reports_by_symbol = load_latest_reports_by_symbol()
    if not reports_by_symbol:
        print("[auto_bot] sharpe: any report detected.", flush=True)
        return

    account = api.get_account()
    cash_total = float(account.cash)

    for symbol, path in reports_by_symbol.items():
        try:
            asset = api.get_asset(symbol)
            if not getattr(asset, "tradable", False):
                print(f"[auto_bot] {symbol}: non tradable -> skip.", flush=True)
                logging.info("%s: non tradable -> skip", symbol)
                continue

            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

            name, sig_col, ret_col, sr = choose_best_strategy_by_sharpe(df)
            if sig_col is None:
                print(f"[auto_bot] {symbol}: any strategy detected -> skip.", flush=True)
                continue
            if (sr is None) or (sr < 0):
                print(f"[auto_bot] {symbol}: Sharpe 30j < 0/insuffisant -> skip.", flush=True)
                continue

            last_signal = int(df[sig_col].dropna().iloc[-1])
            if last_signal == 0:
                print(f"[auto_bot] {symbol}: signal 0 -> nothing to do.", flush=True)
                continue

            last_price = float(api.get_latest_trade(symbol).price)
            try:
                default_qty, _ = determine_qty(symbol, mode="paper", last_price=last_price)
            except TypeError:
                default_qty, _ = determine_qty(symbol, mode="paper")
            qty = max(0, int(default_qty))
            if qty == 0:
                print(f"[auto_bot] {symbol}: qty=0 (cash insuffisant?) -> skip.", flush=True)
                continue

            cost = qty * last_price
            if cost > cash_total:
                print(f"[auto_bot] {symbol}: coût {cost:.2f}$ > cash {cash_total:.2f}$ -> skip.", flush=True)
                continue

            if var_dollars > 0 and not can_place_order(cost, k_pct=K_PCT, portfolio_var_dollars=var_dollars):
                budget = (K_PCT / 100.0) * abs(var_dollars)
                print(f"[auto_bot] {symbol}: BLOCKED k%×VaR (coût {cost:.0f}$ > budget {budget:.0f}$).", flush=True)
                continue

            side = "buy" if last_signal == 1 else "sell"
            api.submit_order(symbol=symbol, qty=qty, side=side, type="market", time_in_force="gtc")
            if var_dollars > 0:
                reserve_order_risk(cost)
            print(f"[auto_bot] {symbol}: {side.upper()} {qty} @ ~{last_price:.2f} OK (Sharpe30j={sr:.2f})", flush=True)
        except Exception as e:
            print(f"[auto_bot] {symbol}: erreur {e}", flush=True)
            logging.exception("%s: error: %s", symbol, e)

def run_once():
    if MODE == "rebalance":
        print("[auto_bot] mode: rebalance (memo/poids).")
        run_rebalance_once()
    else:
        print("[auto_bot] mode: sharpe (sans poids memo).")
        run_sharpe_once()

def main():
    print_config()
    is_open = market_open_now_utc()
    print(f"[auto_bot] market_open_now_utc={is_open}", flush=True)
    logging.info("market_open_now_utc=%s", is_open)
    if LOOP:
        print("[auto_bot] start loop mode (mode={MODE}, deploy={DEPLOY_PCT}%)", flush=True)
        logging.info("start loop mode")
        while True:
            if (not ONLY_MARKET) or market_open_now_utc():
                print("[auto_bot] tick", flush=True)
                logging.info("tick")
                run_once()
            else:
                print("[auto_bot] hors heures de marche -> attente", flush=True)
                logging.info("sleep (market closed)")
            time.sleep(SLEEP_SEC)
    else:
        if (not ONLY_MARKET) or market_open_now_utc():
            print(f"[auto_bot] run once (mode={MODE})", flush=True)
            logging.info("run once")
            run_once()
        else:
            print("[auto_bot] hors heures de marche (execution unique ignoree)", flush=True)
            logging.info("one-shot ignored (market closed)")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print(f"[auto_bot] FATAL: {err}", flush=True)
        logging.exception("FATAL")


