import pandas as pd
import datetime
#import os
#import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backtester import Backtester
from reporting.performance_report import calculate_performance_metrics, generate_verdict


class PortfolioBacktester:
    def __init__(self, data_dict, strategies, allocations=None):
        """
        data_dict : dict => { 'AAPL': df1, 'TSLA': df2, ... }
        strategies : list de tuples => [ ('AAPL', SmaStrategy, params), ... ]
        allocations : dict => { 'AAPL': 0.3, 'TSLA': 0.7 } ou None (égalitaire)
        """
        self.data = data_dict
        self.strategies = strategies
        self.allocations = allocations or {symbol: 1 / len(strategies) for symbol, _, _ in
                                           strategies}  # On extrait seulement le 1er élément de chaque tuple (symbol), et on ignore les 2 autres (_ et _ = convention Python pour "je ne m’en sers pas").
        self.results = {}
        self.portfolio_curve = None

    def run(self):
        equity_curves = []

        for symbol, strategy_cls, params in self.strategies:
            df = self.data[symbol]
            bt = Backtester(df.copy(), strategy_cls, strategy_name=symbol, **params)
            bt.run()
            self.results[symbol] = bt

            alloc = self.allocations.get(symbol, 1 / len(self.strategies))
            weighted_equity = bt.results["Equity_Curve"] * alloc
            equity_curves.append(weighted_equity)

        self.portfolio_curve = pd.concat(equity_curves, axis=1).sum(axis=1)
        return self.portfolio_curve

    def report(self):
        if self.portfolio_curve is None:
            print("Run le portefeuille d'abord.")
            return

        df = pd.DataFrame({
            "Equity_Curve": self.portfolio_curve
        })
        df["Strategy_Returns"] = df["Equity_Curve"].pct_change()

        # Nettoyage
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Rapport
        try:
            perf = calculate_performance_metrics(df, strategy_col="Strategy_Returns", skip_signal=True)
            decision, motifs = generate_verdict(perf, skip_signal=True)

            print("\n=== Rapport Global du Portefeuille ===")
            print(perf)
            print(f"\n✅ Verdict : {decision}")
            print("Motifs :", ", ".join(motifs))

            # Export CSV
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"reporting/rapport_portefeuille_{now}.csv"
            perf.to_csv(filename)
            print(f"\n📁 Rapport enregistré : {filename}")

            return perf

        except Exception as e:
            print(f"❌ Erreur dans le rapport du portefeuille : {e}")
            return perf