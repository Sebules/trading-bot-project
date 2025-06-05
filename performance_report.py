import pandas as pd
import numpy as np


def calculate_performance_metrics(df, strategy_col="Strategy_Returns"):
 """
 Calcule les principaux indicateurs de performance d'une stratégie.
 df : DataFrame contenant les colonnes 'Market_Returns' et strategy_col
 strategy_col : nom de la colonne des rendements de la stratégie
 """

 # Nettoyage
 df = df.copy().dropna()

 # Total Return
 total_return = (1 + df[strategy_col]).prod() - 1

 # Sharpe Ratio (on suppose 252 jours de trading/an)
 excess_returns = df[strategy_col] - 0  # taux sans risque = 0 pour simplifier
 sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

 # Max Drawdown
 equity_curve = (1 + df[strategy_col]).cumprod()
 roll_max = equity_curve.cummax()
 drawdown = (equity_curve - roll_max) / roll_max
 max_drawdown = drawdown.min()

 # Nombre de trades
 n_trades = df["Signal"].diff().fillna(0).abs().sum()

 # Win Rate
 winning_trades = df.loc[df["Signal"].shift(1) == 1, strategy_col] > 0
 win_rate = winning_trades.mean()

 # Volatilité annualisée
 volatility = df[strategy_col].std() * np.sqrt(252)

 report = {
  "Total Return (%)": round(total_return * 100, 2),
  "Sharpe Ratio": round(sharpe, 2),
  "Max Drawdown (%)": round(max_drawdown * 100, 2),
  "Win Rate (%)": round(win_rate * 100, 2),
  "Nombre de Trades": int(n_trades),
  "Volatilité (%)": round(volatility * 100, 2),
 }

 return pd.Series(report)


def generate_verdict(report):
 sharpe = report["Sharpe Ratio"]
 drawdown = report["Max Drawdown (%)"]
 win_rate = report["Win Rate (%)"]
 total_return = report["Total Return (%)"]

 verdict = []

 # Critères d’évaluation
 if sharpe > 1.5:
  verdict.append("Sharpe OK")
 elif sharpe < 0.5:
  verdict.append("Sharpe FAIBLE")

 if drawdown < 0.2:
  verdict.append("Drawdown faible")
 elif drawdown > 0.4:
  verdict.append("Drawdown élevé")

 if win_rate > 0.55:
  verdict.append("Bonne fréquence de gains")
 elif win_rate < 0.45:
  verdict.append("Trop de trades perdants")

 if total_return > 0.10:
  verdict.append("Performance OK")
 elif total_return < 0:
  verdict.append("Retour négatif")

 # Décision finale
 if all("OK" in v or "Bonne" in v or "faible" in v for v in verdict):
  decision = "STRATÉGIE À CONSERVER"
 elif "Retour négatif" in verdict or "Sharpe FAIBLE" in verdict:
  decision = "STRATÉGIE À ÉVITER"
 else:
  decision = "STRATÉGIE À AMÉLIORER"

 return decision, verdict