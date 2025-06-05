import numpy as np
import pandas as pd
def compute_performance(df, strategy_col='Strategy_Returns'):
    report = {}

    # Total Return
    total_return = df[strategy_col].add(1).prod() - 1
    report["Total Return"] = round(total_return * 100, 2)

    # Annualized Volatility
    volatility = df[strategy_col].std() * np.sqrt(252)
    report["Volatility (ann.)"] = round(volatility * 100, 2)

    # Sharpe Ratio
    sharpe = df[strategy_col].mean() / df[strategy_col].std() * np.sqrt(252)
    report["Sharpe Ratio"] = round(sharpe, 2)

    # Max Drawdown
    equity = (1 + df[strategy_col]).cumprod()
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    report["Max Drawdown"] = round(max_drawdown * 100, 2)

    # Win Rate
    wins = df[strategy_col][df[strategy_col] > 0].count()
    total = df[strategy_col].count()
    win_rate = wins / total
    report["Win Rate"] = round(win_rate * 100, 2)

    # Nombre de trades (changements de position)
    if "Signal" in df.columns:
        trades = df["Signal"].diff().fillna(0).ne(0).sum()
        report["Nombre de trades"] = int(trades)

    return report