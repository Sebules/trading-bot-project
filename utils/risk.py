# risk.py
import numpy as np
import pandas as pd

try:
    import requests
except Exception:
    requests = None  # Fichier importable même sans requests

def _to_returns(series: pd.Series) -> pd.Series:
    s = pd.Series(series).dropna()
    # Heuristique simple : equity ($) vs rendements (%)
    if s.abs().median() > 0.3:
        r = s.pct_change().dropna()
    else:
        r = s
    return r.astype(float)

def ex_ante_volatility(returns: pd.Series, annualize: bool = True, periods: int = 252) -> float:
    r = _to_returns(returns)
    sigma = float(r.std())
    return (np.sqrt(periods) * sigma) if annualize else sigma

def var_historic(returns: pd.Series, alpha: float = 0.95) -> float:
    r = _to_returns(returns)
    q = np.quantile(r, 1 - alpha)   # ex: 5e pct si alpha=0.95
    return float(max(0.0, -q))      # magnitude positive

def es_historic(returns: pd.Series, alpha: float = 0.95) -> float:
    r = _to_returns(returns)
    q = np.quantile(r, 1 - alpha)
    tail = r[r <= q]
    return 0.0 if tail.empty else float(-tail.mean())

def var_to_dollars(var_return: float, equity_value: float) -> float:
    return float(abs(equity_value) * max(0.0, var_return))

def max_position_size(last_price: float, portfolio_var_dollars: float, k_pct: float) -> int:
    if last_price <= 0:
        return 0
    budget = (k_pct / 100.0) * abs(portfolio_var_dollars)
    return int(budget // last_price)

def send_to_slack(text: str, webhook_url: str) -> bool:
    if not webhook_url or requests is None:
        return False
    try:
        resp = requests.post(webhook_url, json={"text": text}, timeout=5)
        return resp.ok
    except Exception:
        return False