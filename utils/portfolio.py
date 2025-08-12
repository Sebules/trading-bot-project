import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

def estimate_mu_sigma(returns: pd.DataFrame, span: int = 60):
    """
    Estime les rendements attendus (EWM) et la covariance shrunkée (Ledoit-Wolf).
    returns: DataFrame de rendements journaliers
    span: paramètre pour l'EWM
    """
    # Gestion des valeurs manquantes : forward/backward fill puis suppression résiduelle
    rets = returns.copy()
    rets = rets.fillna(method='ffill').fillna(method='bfill')
    rets = rets.dropna(how='any')
    if rets.empty:
        raise ValueError("Pas assez de données post-imputation pour estimer mu et Sigma")

    # Rendements annualisés via EWM
    mu_ewm = rets.ewm(span=span).mean().iloc[-1] * 252

    # Covariance shrunkée
    lw = LedoitWolf().fit(rets)
    Sigma_shrink = lw.covariance_ * 252

    Sigma_df = pd.DataFrame(
        Sigma_shrink,
        index=returns.columns,
        columns=returns.columns
    )
    return mu_ewm, Sigma_df


def optimize_weights(returns: pd.DataFrame, risk_free_rate: float = 0.0, max_weight: float = 0.3,
                     span_ewm: int = 60) -> pd.Series:
    """
        Calcule les poids optimaux en maximisant le Sharpe, avec:
        - estimation EWM + shrinkage
        - bornes sur poids
        - fallback sur solveur COBYLA si SLSQP échoue
        """
    # 1. Filtrage des actifs à variance non-nulle
    valid = returns.columns[returns.std() > 1e-8]
    if len(valid) <= 1:
        # Fallback vers portefeuille égalitaire 1/N
        w0 = pd.Series(1.0 / len(returns.columns), index=returns.columns)
        return w0.fillna(0)

    rets = returns[valid]

    # 2. Estimation robuste des paramètres
    mu, Sigma = estimate_mu_sigma(rets, span=span_ewm)

    # 3. Définition de la fonction de négation du Sharpe à minimiser
    def neg_sharpe(w):
        port_ret = w.dot(mu)
        port_vol = np.sqrt(w.dot(Sigma).dot(w))
        # penalise une vol=0
        return -(port_ret - risk_free_rate) / port_vol if port_vol > 0 else 1e6

    # 4. Contraintes et bornes
    n = len(valid)
    bounds = tuple((0.0, max_weight) for _ in range(n))  # MOD: bornes max_weight
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    # 5. Initialisation et optimisation
    w0 = np.ones(n) / n
    res = minimize(
        neg_sharpe,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    # MODIFICATION: fallback sur COBYLA si échec
    if not res.success:
        res = minimize(
            neg_sharpe,
            w0,
            method='COBYLA',
            bounds=bounds,
            constraints=constraints
        )

    # 6. Construction de la série finale avec normalisation
    raw_w = res.x
    raw_w = np.clip(raw_w, 0.0, None)  # on retire les valeurs négatives éventuelles
    norm_w = raw_w / raw_w.sum()

    w_opt = pd.Series(0.0, index=returns.columns)
    # Remplissage des poids pour actifs valides
    w_opt.loc[valid] = norm_w
    return w_opt


    """
    Trouve la pondération w qui maximise le Sharpe Ratio :
        Sharpe = (E[R_p] - rf) / σ_p
    sous contraintes :
        sum(w) = 1, w_i >= 0.

    returns : DataFrame de rendements quotidiens (colonnes = actifs)
    rf      : taux sans risque (par défaut 0)

    Retourne une Series pandas des poids optimaux.
    """

def generate_random_portfolios(returns: pd.DataFrame, n_portf: int = 5000):
    np.random.seed(42)
    n = returns.shape[1]
    results = np.zeros((3, n_portf))
    for i in range(n_portf):
        w = np.random.dirichlet(np.ones(n), size=1).flatten()
        ret = (returns.mean() * w).sum() * 252
        vol = np.sqrt(w @ (returns.cov() * 252) @ w)
        sharpe = (ret - 0.0) / vol
        results[:, i] = [vol, ret, sharpe]
    return results

