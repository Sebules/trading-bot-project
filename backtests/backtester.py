import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.performance import compute_performance
from reporting.performance_report import calculate_performance_metrics, generate_verdict

class Backtester:
    def __init__(self, df, strategy_cls, strategy_name="Unnamed", **strategy_params):
        self.df = df.copy()
        self.strategy_cls = strategy_cls  # self.strategy_cls est une classe de stratégie (comme MovingAverageStrategy, RSIStrategy, etc.)
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params
        self.results = None
        self.report = None

    def run(self):
        strategy = self.strategy_cls(self.df, **self.strategy_params)
        df = strategy.generate_signals()

        if 'Close' not in df.columns or 'Signal' not in df.columns:
            raise ValueError("Le DataFrame doit contenir les colonnes 'Close' et 'Signal'.")

        df["Market_Returns"] = df["Close"].pct_change()
        df["Strategy_Returns"] = df["Market_Returns"] * df["Signal"].shift(1)  # On agit sur le signal d'hier
        df["Equity_Curve"] = (1 + df["Strategy_Returns"]).cumprod()

        self.results = df
        return df

    def plot(self):
        if self.results is None:
            print("Run le backtest d'abord.")
            return

        plt.figure(figsize=(14, 6))
        plt.plot(self.results["Close"], label="Prix")
        plt.plot(self.results["Equity_Curve"], label="Performance stratégie")
        plt.title(f"Stratégie : {self.strategy_name}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate(self):
        return compute_performance(self.results)

    def reports(self):
        self.report = calculate_performance_metrics(self.results)
        if self.report is not None:
            print(f"\nRapport de performance pour {self.strategy_name}:\n")
            print(self.report)
            decision, verdicts = generate_verdict(self.report)
            print(f"\nVerdict : {decision}")
            print("Motifs :", ", ".join(verdicts))
            self.report["Strategy"] = self.strategy_name
        return self.report

    def report_csv(self):
        self.report.to_csv(f"reporting/rapport_{self.strategy_name}.csv")
        return

    def _max_drawdown(self, equity_curve):
        roll_max = equity_curve.cummax()
        drawdown = (equity_curve - roll_max) / roll_max
        return drawdown.min()
