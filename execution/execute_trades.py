
import pandas as pd
from alpaca_trade_api.rest import REST


def get_current_position_qty(api: REST, symbol: str) -> int:
    """Retourne la position existante (quantité), sinon 0"""
    try:
        position = api.get_position(symbol)
        return int(float(position.qty))
    except:
        return 0

def execute_signal(api: REST, symbol: str, signal: int, qty: int = 1):
    """Exécute un ordre de marché en fonction du signal"""
    try:
        if signal == 1:
            api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc')
            return f"🟢 Achat exécuté sur {symbol}"
        elif signal == -1:
            api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
            return f"🔴 Vente exécutée sur {symbol}"
        else:
            return f"🟡 Aucune action requise sur {symbol}"
    except Exception as e:
        return f"❌ Erreur exécution {symbol} : {e}"


