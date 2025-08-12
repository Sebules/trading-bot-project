from pathlib import Path
import re
import pandas as pd

_rx = re.compile(r"rapport_([^_]+)_")


def extract_symbol(path: Path) -> str:
    """
    ↳ AAPL à partir de rapport_AAPL_20250716.csv
    """
    m = _rx.match(path.name)
    if not m:
        raise ValueError(f"Nom de fichier non conforme: {path.name}")
    return m.group(1).upper()


def best_strategy_and_signal(path: Path) -> tuple[str, int]:
    """
    ↳ ('MACD', 1)  → stratégie gagnante + dernier signal
    """
    df = pd.read_csv(path)
    signal_cols = [c for c in df.columns if c.endswith("_Signal")]
    if not signal_cols:
        raise ValueError("Aucune colonne *_Signal trouvée")

    # On classe les stratégies sur la dernière valeur de Sharpe
    scores = {}
    for c in signal_cols:
        sharpe_col = f"{c[:-7]}_Sharpe"
        if sharpe_col in df.columns:
            score = df[sharpe_col].iloc[-1]
            if pd.notna(score):
                scores[c] = score
    if not scores:
        raise RuntimeError(f"Aucun Sharpe valide dans {path.name}")

    best_col = max(scores, key=scores.get)
    signal = int(df[best_col].iloc[-1])           # 1 / ‑1 / 0
    return best_col[:-7], signal