# broker/metrics.py

import numpy as np
import pandas as pd

def sharpe_from_returns(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna().astype(float)
    if r.empty:
        return 0.0
    mu = r.mean() * periods_per_year
    sig = r.std(ddof=1) * np.sqrt(periods_per_year)
    return 0.0 if sig == 0 or np.isnan(sig) else float(mu / sig)

def rolling_sharpe_30d(price: pd.Series) -> float:
    """price: Série de clôtures. Renvoie le Sharpe des 30 derniers jours."""
    rets = price.pct_change()
    return sharpe_from_returns(rets.tail(30))

def sharpe_ratio_30j(df, return_col):
    """Calcule le Sharpe Ratio sur 30 derniers jours."""
    recent = df[return_col].dropna().tail(30)
    if len(recent) < 2:
        return None
    mean_ret = recent.mean()
    vol = recent.std()
    if vol == 0:
        return None
    return mean_ret / vol

