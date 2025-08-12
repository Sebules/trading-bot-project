import json, os
from datetime import date

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

RISK_FILE = os.path.join(LOGS_DIR,"daily_risk.json")

# S'assurer que le dossier existe
os.makedirs(os.path.dirname(RISK_FILE), exist_ok=True)

# Créer le fichier avec un objet JSON vide, s'il n'existe pas
if not os.path.isfile(RISK_FILE):
    with open(RISK_FILE, "w") as f:
        json.dump({}, f, indent=2)

def _load_daily_risk_used() -> float:
    """Retourne le risque déjà utilisé aujourd'hui (en unité monétaire)."""
    today = date.today().isoformat()
    if not os.path.isfile(RISK_FILE):
        return 0.0
    with open(RISK_FILE, "r") as f:
        data = json.load(f)
    return float(data.get(today, 0.0))

def _save_daily_risk_used(amount: float):
    """Ajoute `amount` au risque utilisé aujourd'hui."""
    today = date.today().isoformat()
    # charge l’existant
    data = {}
    if os.path.isfile(RISK_FILE):
        with open(RISK_FILE, "r") as f:
            data = json.load(f)
    # incrémente
    data[today] = float(data.get(today, 0.0)) + amount
    # écriture atomique
    os.makedirs(os.path.dirname(RISK_FILE), exist_ok=True)
    with open(RISK_FILE, "w") as f:
        json.dump(data, f, indent=2)

# === Risk dashboard helpers (s'appuie sur risk.py) ===
from typing import Dict, Optional
import pandas as pd

try:
    from risk import (
        ex_ante_volatility, var_historic, es_historic,
        var_to_dollars, max_position_size, send_to_slack
    )
except ImportError:
    # si tu ranges risk.py ailleurs (ex: utils/risk.py), adapte cet import
    from utils.risk import (
        ex_ante_volatility, var_historic, es_historic,
        var_to_dollars, max_position_size, send_to_slack
    )

def compute_dashboard_metrics(equity_series, equity_now: float, lookback: int = 60, alpha: float = 0.95) -> Dict[str, float]:
    """
    Calcule Volatilité ex-ante (annualisée), VaR et ES (1j) en % et en $.
    Retourne un dict: vol_ann, var_r, es_r, var_$, es_$.
    """
    s = pd.Series(equity_series).dropna()
    rets = s.pct_change().dropna().tail(lookback)
    if rets.empty or equity_now is None:
        return {"vol_ann": float("nan"), "var_r": float("nan"), "es_r": float("nan"), "var_$": 0.0, "es_$": 0.0}
    vol_ann = ex_ante_volatility(rets, annualize=True)
    var_r = var_historic(rets, alpha=alpha)
    es_r = es_historic(rets, alpha=alpha)
    var_d = var_to_dollars(var_r, float(equity_now))
    es_d = var_to_dollars(es_r, float(equity_now))
    return {"vol_ann": vol_ann, "var_r": var_r, "es_r": es_r, "var_$": var_d, "es_$": es_d}

def allowed_max_qty(last_price: float, portfolio_var_dollars: float, k_pct: float) -> int:
    """Nombre max d'actions pour que l'expo reste <= k% de la VaR portefeuille."""
    return max_position_size(last_price, portfolio_var_dollars, k_pct)

def can_place_order(notional_cost: float, k_pct: float, portfolio_var_dollars: float) -> bool:
    """
    Vrai si l'ordre (en $) tient dans le budget de risque du jour:
    budget = k% × VaR_portefeuille($), puis on soustrait le risque déjà 'utilisé' aujourd'hui.
    """
    budget = (k_pct / 100.0) * abs(float(portfolio_var_dollars))
    used_today = _load_daily_risk_used()
    return float(notional_cost) <= max(0.0, budget - used_today)

def reserve_order_risk(notional_cost: float):
    """Réserve (enregistre) le coût de l'ordre comme risque 'utilisé' aujourd'hui."""
    _save_daily_risk_used(float(notional_cost))

def post_slack_risk_summary(webhook_url: str, metrics: Dict[str, float],
                            symbol: Optional[str] = None,
                            qty_max: Optional[int] = None,
                            k_pct: Optional[float] = None) -> bool:
    """
    Envoie un résumé Risk dashboard dans Slack (retourne True/False).
    """
    try:
        vol_p = f"{metrics['vol_ann']*100:.2f}%" if pd.notna(metrics.get("vol_ann")) else "n/a"
        text = (
            "Risk Dashboard\n"
            f"Vol ann: {vol_p}\n"
            f"VaR 95% (1j): -${metrics['var_$']:,.0f}\n"
            f"ES  95% (1j): -${metrics['es_$']:,.0f}"
        )
        if symbol and qty_max is not None and k_pct is not None:
            text += f"\nMax position {symbol}: {qty_max} (k={int(k_pct)}%)"
        return send_to_slack(text, webhook_url)
    except Exception:
        return False