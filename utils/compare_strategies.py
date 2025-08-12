import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backtests.backtester import Backtester
from reporting.performance_report import generate_verdict

import re

# Nettoyage des noms pour éviter les caractères invalides dans le nom de fichier
def clean_name(name):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name)


def compare_strategies(df, strategies):
    """
    Compare plusieurs stratégies sur un même DataFrame.

    Args:
        df (pd.DataFrame): Données du marché avec colonnes 'Close', etc.
        strategy_classes (list): Liste de classes de stratégies à tester.

    Returns:
        pd.DataFrame: Résumé des performances (Sharpe, Total Return, Win Rate, etc.)
    """
    results = []

    for name, strategy_cls, params in strategies:
        print(f"\n⏳ Test de : {name}")
        bt = Backtester(df, strategy_cls, name, **params)
        try:
            bt.run()
            report = bt.reports()
            # report['Strategy'] = name
            results.append(report)

        except Exception as e:
            print(f"❌ Erreur avec la stratégie {name}: {e}")

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index("Strategy")

    # Classement automatique par Sharpe Ratio
    comparison_df["Rank"] = comparison_df["Sharpe Ratio"].rank(ascending=False, method="min").astype(int)
    comparison_df = comparison_df.sort_values("Rank")
    comparison_df = comparison_df[["Rank"] + [col for col in comparison_df.columns if col != "Rank"]]

    # Ajout d'une colonne Verdict et Motifs
    comparison_df["Verdict"] = comparison_df.apply(lambda row: generate_verdict(row)[0], axis=1)
    comparison_df["Motifs"] = comparison_df.apply(lambda row: ", ".join(generate_verdict(row)[1]), axis=1)

    # Création du dossier s'il n'existe pas
    # os.makedirs("reporting", exist_ok=True)

    # Création du nom de fichier à partir des noms de stratégies
    strategy_names = [clean_name(name).replace(" ", "").lower() for name, _, _ in strategies]
    base_name = "_".join(strategy_names)[:60]

    # Ajout de l'horodatage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"reporting/comparaison_{base_name}_{timestamp}.csv"

    comparison_df.to_csv(filename)
    print(f"\n ***************** \nFichier CSV généré :\n {filename}\n*****************\n")
    print("\n📊 Tableau de comparaison des stratégies :\n")
    print(comparison_df)
    return