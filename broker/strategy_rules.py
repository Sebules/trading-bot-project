# broker/strategy_rules.py

from broker.metrics import sharpe_ratio_30j

def should_switch_strategy(current: dict | None,
                           challenger: dict,
                           alpha: float = 0.10) -> bool:
    """
    Règle lisible:
    - si Sharpe_30j du courant <= 0  → switch
    - sinon, switch si le challenger a un Sharpe_30j supérieur d'au moins alpha (10%)
      ET un max_drawdown <= au courant
    current/challenger attendus: {"name":"RSI","sharpe_30d":0.7,"max_dd":-0.12}
    """
    if current is None:
        return True  # pas de stratégie en place → on prend le challenger

    s_cur, s_new = current.get("sharpe_30d", 0), challenger.get("sharpe_30d", 0)
    dd_cur = current.get("max_dd", 0)
    dd_new = challenger.get("max_dd", 0)

    if s_cur <= 0:
        return True
    return (s_new > s_cur * (1 + alpha)) and (dd_new <= dd_cur)


def choose_strategy(previous_best: dict | None, candidates: list[dict]) -> dict:
    """
    candidates: liste de métriques pour chaque stratégie
    ex: [{"name":"RSI","sharpe_30d":0.6,"max_dd":-0.1}, {"name":"MA","sharpe_30d":0.75,"max_dd":-0.12}]
    Retourne le dict retenu (avec "name").
    """
    # meilleur challenger par Sharpe_30d
    challenger = max(candidates, key=lambda d: d.get("sharpe_30d", -1e9))
    if should_switch_strategy(previous_best, challenger):
        return challenger
    return previous_best

def choose_best_strategy_by_sharpe(df):
    """
    Parcourt les paires (<Strat>_Returns, <Strat>_Signal) présentes dans df
    et retourne la meilleure stratégie selon le Sharpe 30j.
    Retour: (strategy_name, signal_col, returns_col, sharpe)
    - Si au moins une stratégie a Sharpe >= 0, on renvoie la meilleure >= 0.
    - Sinon on renvoie la meilleure (même si < 0).
    - Si rien de détectable, retourne (None, None, None, None).
    """
    candidates = []
    for col in df.columns:
        if col.endswith("_Returns"):
            base = col[:-8]  # retire le suffixe "_Returns"
            signal_col = f"{base}_Signal"
            if signal_col in df.columns:
                sr = sharpe_ratio_30j(df, col)
                candidates.append((base, signal_col, col, sr))

    # Cas générique éventuel ("Returns" / "Signal")
    if not candidates and ("Returns" in df.columns and "Signal" in df.columns):
        sr = sharpe_ratio_30j(df, "Returns")
        candidates.append(("Generic", "Signal", "Returns", sr))

    if not candidates:
        return (None, None, None, None)

    # Sépare celles avec un Sharpe calculable
    valid = [c for c in candidates if c[3] is not None]
    if not valid:
        # pas de Sharpe calculable, on renvoie la première telle quelle
        return candidates[0]

    # Meilleure au sens du Sharpe (descendant)
    best_overall = sorted(valid, key=lambda x: x[3], reverse=True)[0]
    # Tente de privilégier une >= 0
    non_negative = [c for c in valid if c[3] >= 0]
    if non_negative:
        best_non_neg = sorted(non_negative, key=lambda x: x[3], reverse=True)[0]
        return best_non_neg
    return best_overall