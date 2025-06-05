from backtester import Backtester

def run_multiple_backtests(df, strategies):
    """
    Paramètres :
        - df : DataFrame de données (OHLCV)
        - strategies : liste de tuples (nom, classe_stratégie, paramètres)

    Retour :
        - Liste de dicts avec les résultats de chaque backtest
    """
    results = []

    for name, strategy_cls, params in strategies:
        print(f"\n⏳ Test de : {name}")
        bt = Backtester(df, strategy_cls, name, **params)
        try:
            bt.run()
            bt.plot()
            res = bt.evaluate()
            results.append(res)


        except Exception as e:
            print(f"❌ Erreur pour {name} : {e}")
            continue

    return results