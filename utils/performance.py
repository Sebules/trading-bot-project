import numpy as np
import pandas as pd
def compute_performance(df, strategy_col='Strategy_Returns'):
    report = {}

    # Total Return
    total_return = df[strategy_col].add(1).prod() - 1
    report["Total Return (%)"] = round(total_return * 100, 2)

    # Annualized Volatility
    volatility = df[strategy_col].std() * np.sqrt(252)
    report["Volatility (%)"] = round(volatility * 100, 2)

    # Sharpe Ratio
    try:
        returns = df[strategy_col].dropna()
        if returns.empty or returns.std() == 0:
            sharpe = np.nan
        else:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
    except Exception:
        sharpe = np.nan

    report["Sharpe Ratio"] = round(sharpe, 2)

    # Max Drawdown
    equity = (1 + df[strategy_col]).cumprod()
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    report["Max Drawdown (%)"] = round(max_drawdown * 100, 2)

    # Win Rate
    wins = df[strategy_col][df[strategy_col] > 0].count()
    total = df[strategy_col].count()
    win_rate = wins / total
    report["Win Rate (%)"] = round(win_rate * 100, 2)

    # Nombre de trades (changements de position)
    if "Signal" in df.columns:
        trades = df["Signal"].diff().fillna(0).ne(0).sum()
        report["Number of Trades"] = int(trades)

    # Extraction des retours par trade
    # On considère qu'un trade démarre lorsque Signal passe de 0 à 1, et se termine de 1 à 0
    sig = df['Signal']
    entries = df.index[sig.diff() == 1].tolist()
    exits = df.index[sig.diff() == -1].tolist()
    # Si on termine en position, on utilise la dernière date pour clôturer
    if len(exits) < len(entries):
        exits.append(df.index[-1])

    trade_returns = []
    for start, end in zip(entries, exits):
        ret = (1 + df.loc[start:end, strategy_col]).prod() - 1
        trade_returns.append(ret)

    # Nombre de trades
    n_trades2 = len(trade_returns)

    # Profit Factor = somme des profits / somme des pertes (en valeur absolue)
    profits = sum(r for r in trade_returns if r > 0)
    losses = abs(sum(r for r in trade_returns if r < 0))
    profit_factor = profits / losses if losses > 0 else np.nan
    report["Profit Factor"] = round(profit_factor, 2)
    return report