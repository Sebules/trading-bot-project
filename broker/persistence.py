# broker/persistence.py
from pathlib import Path
import pandas as pd
import sqlite3, json

DATA_DIR = Path(__file__).resolve().parent.parent / "data_best_strategies"
DATA_DIR.mkdir(exist_ok=True)

_JSON_PATH   = DATA_DIR / "best_strat.json"
_SQLITE_PATH = DATA_DIR / "bot_state.db"
_TABLE       = "best_strategy"


# ---------- JSON (léger, lisible) ----------
def save_best_strat_json(df: pd.DataFrame, path: Path | None = None) -> None:
    """
    Sauvegarde le DataFrame au format JSON « table ».
    Si aucun chemin n’est fourni, on utilise la constante _JSON_PATH.
    """
    # 1) Déterminer le chemin
    if path is None:
        path = _JSON_PATH

    # 2) S’assurer que le dossier existe
    path.parent.mkdir(parents=True, exist_ok=True)

    # 3) Convertir le DataFrame en chaîne JSON
    json_str = df.to_json(orient="table", indent=2)

    # 4) Ouvrir le fichier et écrire la chaîne
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)

def load_best_strat_json(path: Path | None = None) -> pd.DataFrame | None:
    """
    Charge un DataFrame depuis un fichier JSON au format « table ».
    Si le fichier n’existe pas, renvoie None.
    """
    # 1) Choisir le chemin
    if path is None:
        path = _JSON_PATH

    # 2) Vérifier l’existence du fichier
    if not path.exists():
        return None  # ou éventuellement: raise FileNotFoundError(path)

    # 3) Lire le fichier et renvoyer le DataFrame
    df = pd.read_json(path, orient="table")
    return df


# ---------- SQLite (plus robuste) ----------
def save_best_strat_sqlite(df: pd.DataFrame, db_path: Path | None = None) -> None:
    """
    Sauvegarde un DataFrame dans une base SQLite.
    Remplace la table si elle existe déjà.
    """
    # 1) Choisir le fichier .db
    if db_path is None:
        db_path = _SQLITE_PATH

    # 2) S’assurer que le dossier existe
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 3) Ouvrir la connexion, écrire, commit, fermer
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(_TABLE, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()




def _ensure_table(conn):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {_TABLE}(
            asof     TEXT,
            ticker   TEXT,
            strategy TEXT,
            weight   REAL,
            PRIMARY KEY (asof, ticker)
        );
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_asof ON {_TABLE}(asof);")

def upsert_best_strat_sqlite(df: pd.DataFrame, db_path: Path | None = None):
    """
    Ajoute (ou met à jour) le snapshot 'asof' dans la base SQLite.
    Le DataFrame doit contenir les colonnes: Ticker, Strategy, weight, asof.
    """
    db = db_path or _SQLITE_PATH
    with sqlite3.connect(db) as conn:
        _ensure_table(conn)
        # on convertit en liste de tuples [(date, ticker, strat, weight), ...]
        df["asof"] = df["asof"].dt.strftime("%Y-%m-%d_%H-%M-%S")
        rows = df[["asof", "Ticker", "Strategy", "weight"]].itertuples(index=False)
        conn.executemany(
            f"""INSERT OR REPLACE INTO {_TABLE}
                (asof, ticker, strategy, weight)
                VALUES (?, ?, ?, ?);""",
            rows
        )
        conn.commit()

def load_best_strat_sqlite(db_path: Path | None = None,
                           date: str | None = None) -> pd.DataFrame | None:
    """
    - date=None  ➜ charge le snapshot le plus récent
    - date="YYYY-MM-DD" ➜ charge ce jour-là
    """
    db = db_path or _SQLITE_PATH
    if not Path(db).exists():
        return None

    with sqlite3.connect(db) as conn:
        _ensure_table(conn)
        if date is None:
            query = f"""
                SELECT * FROM {_TABLE}
                WHERE asof = (SELECT MAX(asof) FROM {_TABLE});
            """
            return pd.read_sql(query, conn)
        else:
            return pd.read_sql(f"SELECT * FROM {_TABLE} WHERE asof = ?", conn, params=(date,))