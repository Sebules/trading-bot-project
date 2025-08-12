# alpaca_executor.py
from pathlib import Path
try:
    from alpaca.trading.client import TradingClient
except Exception:
    TradingClient = None
import pandas as pd
from utils.report_utils import extract_symbol, best_strategy_and_signal
import logging
import matplotlib.pyplot as plt
import os, io

logger = logging.getLogger(__name__)

def get_position_qty(client: TradingClient, symbol: str) -> int:
    try:
        pos = next(p for p in client.get_all_positions() if p.symbol == symbol)
        return int(pos.qty)
    except StopIteration:
        return 0


def determine_qty(
    client: TradingClient,
    symbol: str,
    price: float,
    side: str,
    risk_pct: float = 0.01,
) -> int:
    """
    - Risque max = risk_pct * equity
    - Jamais plus que le buying_power disponible
    - Si close ou short, on ne dépasse pas la position existante
    """
    acct = client.get_account()
    equity = float(acct.equity)
    buying_power = float(acct.buying_power)

    risk_budget = equity * risk_pct
    qty_by_risk = int(risk_budget // price)
    qty_by_cash = int(buying_power // price)
    qty = max(1, min(qty_by_risk, qty_by_cash))

    # ⚠️  Cas SELL : ne pas vendre plus que l'on détient
    if side == "sell":
        qty = min(qty, get_position_qty(client, symbol))

    return qty


def execute_best_signals(
    client: TradingClient,
    reports_dir: Path,
    risk_pct: float = 0.01,
) -> int:
    """
    Parcourt tous les rapports_<SYMBOL>_YYYYMMDD.csv du répertoire,
    choisit la meilleure stratégie, et envoie les ordres.
    Retourne le nombre d’ordres exécutés.
    """
    n_orders = 0
    for path in reports_dir.glob("rapport_*.csv"):
        symbol = extract_symbol(path)
        try:
            strat, signal = best_strategy_and_signal(path)
        except (RuntimeError, ValueError) as err:
            logger.warning("Rapport ignoré (%s) : %s", path.name, err)
            continue

        if signal == 0:          # Pas de trade
            continue

        side = "buy" if signal > 0 else "sell"
        price = float(client.get_latest_trade(symbol).price)

        qty = determine_qty(client, symbol, price, side, risk_pct)
        if qty == 0:
            continue            # Pas de cash ou position insuffisante

        client.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
            client_order_id=f"{symbol}-{strat}-{pd.Timestamp.utcnow().isoformat()}",
        )
        n_orders += 1
    return n_orders

def plot_equity_png(df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 3), dpi=200)
    ax.plot(df.index, df["equity"])
    ax.set_title("Your portfolio")
    ax.set_ylabel("Equity")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
