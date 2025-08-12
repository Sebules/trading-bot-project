# backtests/backtest_PnL.py
import pandas as pd

def calculer_pnl_depuis_csv(path_csv):
    """
    Calcule le PnL (Profit and Loss) total à partir d'un CSV contenant :
    colonnes: Date, Side ('buy'/'sell'), Prix, Quantité.

    Rappels:
    - PnL (Profit and Loss) = somme des gains/pertes.
    - On ferme toute position restante au dernier prix si besoin.
    """
    df = pd.read_csv(path_csv)
    cash = 0.0
    position = 0
    last_price = None

    for _, row in df.iterrows():
        side = str(row["Side"]).lower()
        price = float(row["Prix"])
        qty = int(row["Quantity"])
        last_price = price

        if side == "buy":
            cash -= price * qty
            position += qty
        elif side == "sell":
            cash += price * qty
            position -= qty
        else:
            raise ValueError(f"Side invalide: {row['Side']} (attendu: 'buy' ou 'sell')")

    # Si position non close, on ferme au dernier prix observé
    if position != 0 and last_price is not None:
        cash += position * last_price
        position = 0

    return float(cash)