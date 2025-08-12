# broker/cash.py
def compute_investable_cash(account_cash: float, pct_to_use: float) -> float:
    """
    Calcule le budget réellement mobilisable pour de nouveaux ordres.

    Parameters
    ----------
    account_cash : float
        Solde espèces du compte (USD).
    pct_to_use : float
        Pourcentage du cash que l’on veut engager (0–100).

    Returns
    -------
    float
        Montant, en dollars, que l’on est autorisé à investir.
    """
    return account_cash * (pct_to_use / 100.0)