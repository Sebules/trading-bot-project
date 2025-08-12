import pandas as pd
import numpy as np


def calculate_performance_metrics(df, strategy_col="Strategy_Returns"):
 """
 Calcule les principaux indicateurs de performance d'une stratégie.
 df : DataFrame contenant les colonnes 'Market_Returns' et strategy_col
 strategy_col : nom de la colonne des rendements de la stratégie
 """

 # Nettoyage
 df = df.copy()
 if df.empty:
  return pd.Series({metric: 0.0 for metric in
                    ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)", "Number of Trades",
                     "Profit Factor", "Volatilité (%)", "Calmar Ratio"]})

 # Total Return
 equity_curve = (1 + df[strategy_col]).cumprod()
 total_return = equity_curve.iloc[-1] - 1
 annualized_return = equity_curve.iloc[-1] ** (252 / len(df)) - 1

 # Sharpe Ratio (on suppose 252 jours de trading/an)
 volatility = (df[strategy_col] -0).std() * np.sqrt(252)  # taux sans risque = 0 pour simplifier
 sharpe = annualized_return / volatility if volatility > 0 else 0

 # Max Drawdown
 roll_max = equity_curve.cummax()
 drawdown = (equity_curve - roll_max) / roll_max
 max_drawdown = drawdown.min()
 calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0

 # Nombre de trades
 n_trades = df["Signal"].diff().fillna(0).abs().sum()/2 if "Signal" in df.columns else np.nan

 # Win Rate
 winning_trades = df[df[strategy_col] > 0]
 losing_trades = df[df[strategy_col] < 0]
 win_rate = len(winning_trades) / (len(winning_trades)
                                   + len(losing_trades)) if (len(winning_trades) + len(losing_trades)) > 0 else 0

 # --- Comparaison Benchmark ---
 # On suppose que 'Close' est dans le DataFrame pour le Buy & Hold
 if 'Close' in df.columns:
  buy_hold_returns = df['Close'].pct_change().fillna(0)
  buy_hold_total_return = (1 + buy_hold_returns).prod() - 1
  perf_vs_bh = total_return - buy_hold_total_return
 else:
  perf_vs_bh = np.nan

 # Profit Factor = somme des profits / somme des pertes (en valeur absolue)
 profits = winning_trades[strategy_col].sum()
 losses = abs(losing_trades[strategy_col].sum())
 profit_factor = profits / losses if losses > 0 else np.nan

 # Volatilité annualisée
 volatility = df[strategy_col].std() * np.sqrt(252)

 report = {
  "Total Return (%)": round(total_return * 100, 2),
  "Sharpe Ratio": round(sharpe, 2),
  "Calmar Ratio": round(calmar_ratio, 2),
  "Max Drawdown (%)": round(max_drawdown * 100, 2),
  "Win Rate (%)": round(win_rate * 100, 2),
  "Number of Trades": int(n_trades),
  #"Performance vs Buy & Hold (%)": round(perf_vs_bh * 100, 2),
  "Profit Factor": round(profit_factor, 2),
  "Volatilité (%)": round(volatility * 100, 2),
 }

 return pd.Series(report)


def generate_verdict(report):
 """Génère un verdict basé sur un système de scoring."""
 score = 0
 verdict = []

 # Score basé sur le Sharpe Ratio
 sharpe = report["Sharpe Ratio"]
 if sharpe > 3:
  score += 3
  verdict.append(f"Excellent Sharpe Ratio ({sharpe})")
 elif sharpe > 2:
  score += 2
  verdict.append(f"Bon Sharpe Ratio ({sharpe})")
 elif sharpe > 1:
  score += 1
  verdict.append(f"Sharpe Ratio acceptable ({sharpe})")
 else:
  score -= 1
  verdict.append(f"Faible Sharpe Ratio ({sharpe})")

 # Score basé sur le Calmar Ratio
 calmar = report["Calmar Ratio"]
 if calmar > 3:
  score += 2
  verdict.append(f"Excellent Calmar Ratio ({calmar})")
 elif calmar > 1:
  score += 1
  verdict.append(f"Raisonnable Calmar Ratio ({calmar})")
 else :
  score -= 2
  verdict.append(f"Faible Calmar Ratio ({calmar})")

 # Pénalité pour Drawdown élevé
 drawdown = report["Max Drawdown (%)"]
 if drawdown < -30:
  score -= 2
  verdict.append(f"Drawdown très élevé ({drawdown}%)")
 elif drawdown < -15:
  score -= 1
  verdict.append(f"Drawdown modéré ({drawdown}%)")

 # Bonus/Malus sur la performance vs Buy & Hold
 #perf_vs_bh = report["Performance vs Buy & Hold (%)"]
 # if isinstance(perf_vs_bh, (int, float)):
 #  if perf_vs_bh > 5:
 #   score += 2
 #   verdict.append(f"Surperformance notable vs Buy & Hold ({perf_vs_bh:.1f}%)")
 #  elif perf_vs_bh < 0:
 #   score -= 2
 #   verdict.append(f"Sous-performance vs Buy & Hold ({perf_vs_bh:.1f}%)")


 win_rate = report["Win Rate (%)"]
 total_return = report["Total Return (%)"]
 num_trades = report["Number of Trades"]

 # Score basé sur le Facteur de Profit
 profit_factor = report["Profit Factor"]
 if profit_factor > 2:
  score += 2
  verdict.append(f"Excellent Facteur de Profit ({profit_factor:.2f})")
 elif profit_factor > 1.5:
  score += 1
  verdict.append(f"Bon Facteur de Profit ({profit_factor:.2f})")
 elif profit_factor <= 1.5:
  score += 0
  verdict.append(f"Facteur de Profit Passable ({profit_factor:.2f})")
 elif profit_factor < 1:
  score -= 2
  verdict.append(f"Facteur de Profit < 1 ({profit_factor:.2f})")




 # Critères d’évaluation
 if win_rate > 55:
  verdict.append("Bonne fréquence de gains")
 elif win_rate < 45:
  verdict.append("Trop de trades perdants")

 if total_return >= 10:
  verdict.append("Performance OK")
 elif total_return < 0:
  verdict.append("Retour négatif")

 if not np.isnan(num_trades):
  if num_trades < 60:
   verdict.append("Attention! Trop peu de trades si stratégie à court terme")
  elif num_trades > 1000:
   verdict.append("Beaucoup de trades")
  else:
   verdict.append("Nombre de trades OK")


 # Décision finale
 # if all("OK" in v or "Bonne" in v or "faible" in v for v in verdict):
 #  decision = "STRATÉGIE À CONSERVER"
 # elif any(k in "|".join(verdict) for k in ["Retour négatif",
 #                                           "Sharpe FAIBLE",
 #                                           "Facteur de profit FAIBLE",
 #                                           "Trop peu de trades"]):
 #  decision = "STRATÉGIE À ÉVITER"
 # else:
 #  decision = "STRATÉGIE À AMÉLIORER"


 # Décision finale basée sur le score
 if score >= 3:
  decision = "STRATÉGIE À CONSERVER"
 elif score <= 0:
  decision = "STRATÉGIE À ÉVITER"
 else:
  decision = "STRATÉGIE À AMÉLIORER"

 return decision, verdict, score
