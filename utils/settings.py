# utils/settings.py
from pathlib import Path
import os
from dotenv import load_dotenv
from pathlib import Path
import os

def _get_from_streamlit(key):
    try:
        import streamlit as st
        # On essaie d'accéder prudemment; si pas de secrets.toml, on tombe en except
        return st.secrets.get(key)
    except Exception:
        return None

def _get(key, default=None):
    # 1) si on est dans Streamlit (et secrets dispos), on prend st.secrets
    val = _get_from_streamlit(key)
    if val:
        return val
    # 2) sinon on prend la variable d'environnement
    env = os.getenv(key)
    if env:
        return env
    # 3) fallback
    return default

# --- API Alpaca ---
ALPACA_API_KEY    = _get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = _get("ALPACA_SECRET_KEY")
ALPACA_BASE_URL   = _get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_PAPER_URL = _get("ALPACA_PAPER_URL")
ALPACA_PAPER = _get("ALPACA_PAPER")

#--- AUTO BOT ---
RISK_K_PCT = _get("RISK_K_PCT")
AUTO_BOT_SLEEP = _get("AUTO_BOT_SLEEP")
AUTO_BOT_LOOP = _get("AUTO_BOT_LOOP")
AUTO_BOT_ONLY_MARKET = _get("AUTO_BOT_ONLY_MARKET")

#--- Slack ---
SLACK_WEBHOOK_URL = _get("SLACK_WEBHOOK_URL")

#--- API ALPHA VANTAGE ---
ALPHA_API_KEY = _get("AV_API_KEY")

# --- Racine data (commune à Streamlit et aux scripts cron) ---
DATA_ROOT = Path(_get("BOT_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
REPORT_ROOT = Path(_get("BOT_REPORT_DIR", Path(__file__).resolve().parent.parent / "reporting"))
RESULT_ROOT = Path(_get("BOT_RESULT_DIR", REPORT_ROOT / "resultat"))
ML_TRAIN_ROOT= Path(_get("BOT_ML_DIR", Path(__file__).resolve().parent.parent / "ml"))
ML_MODELS_ROOT= Path(_get("BOT_MODELS_DIR", ML_TRAIN_ROOT / "trained_models"))

DATA_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
ML_TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
ML_MODELS_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_CASH_DEPLOY_PCT = float(_get("DEFAULT_CASH_PCT", 25.0))