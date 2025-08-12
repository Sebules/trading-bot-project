# broker/daily_task.py

from broker.persistence import load_best_strat_json

previous_best = load_best_strat_json()
if previous_best is not None:
    # ex : éviter de recalculer une nouvelle stratégie si Sharpe n’a pas bougé
    cached = {row["Symbol"]: row["Strategy"] for _, row in previous_best.iterrows()}
else:
    cached = {}

