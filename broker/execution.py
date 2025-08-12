# broker/execution.py
from broker.cash import compute_investable_cash

def build_order_plan(api, positions, memo: dict, cash: float, pct: float, return_recap=False):
    """
    Construit la liste des ordres à envoyer sans aucun affichage Streamlit.
    Retourne une liste de dictionnaires {symbol, side, qty, est_cost}.
    """
    budget_total = compute_investable_cash(cash, pct)
    plan, recap = [], []

    all_syms = set(memo.keys()) | set(positions.keys())
    THRESH = 0.0025  # 0,25 %
    # total_portfolio_value (actions + cash) pour un gap “juste”
    equity_values = [
        int(positions.get(sym, 0)) * float(api.get_latest_trade(sym).price)
        for sym in positions
    ]
    total_equity = sum(equity_values) + cash  # include cash

    for symbol in all_syms:
        if symbol in memo:
            strat, weight_pct = memo[symbol]  # dépaquetage
        else:
            strat, weight_pct = None, 0         # orphelin ➜ poids 0 %

        w            = weight_pct / 100.0
        target_value = total_equity * w
        price        = float(api.get_latest_trade(symbol).price)
        curr_qty     = int(positions.get(symbol, 0))
        target_qty   = int(target_value // price)
        float_delta = target_value / price - curr_qty
        delta        = target_qty - curr_qty

        # seuil sous-pondération / sur-pondération
        weight_gap = abs(target_value - curr_qty * price) / total_equity
        if delta == 0 and weight_gap > THRESH:
            delta = 1 if float_delta > 0 else -1

        recap.append({
            "Symbol": symbol,
            "Poids (%)": round(weight_pct, 3),
            "Prix": price,
            "Qty actuelle": curr_qty,
            "Qty cible": target_qty,
            "Δ Qty": delta,
            "Valeur cible $": round(target_qty * price, 2),
        })

        if delta != 0:
            plan.append({
                "symbol": symbol,
                "side": "buy" if delta > 0 else "sell",
                "qty": abs(delta),
                "est_cost": abs(delta) * price
            })
    return (plan, recap) if return_recap else plan