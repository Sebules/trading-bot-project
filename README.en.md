# 📊 Trading Algo Dashboard (Streamlit)

A **multi‑tool** dashboard for market analysis, strategy comparison, ML, risk management, and **(paper) execution** via Alpaca.

> ⚠️ **Disclaimer** — For educational purposes only. This is *not* financial advice. Test in **paper trading** first.

---

## 🗂️ CURRENT repository layout

```
.
├── dashboard.py      # Main dashboard (data loading, strategies, comparisons, charts)
├── 1_Robots.py       # Robots: ML (RF/SVC), risk clustering, portfolio optimization
├── 2_Alpaca.py       # Alpaca integration: account/positions, Risk Dashboard (VaR/ES), order routing
└── (Python packages required by imports; see below)
```

The code expects internal packages (present in your repo even if not attached here):

```
broker/               # cash, Alpaca execution, rules, persistence, metrics, etc.
execution/            # signal execution, run_bot, strategy optimizer
ml/                   # model IO/training; pickles & metadata
reporting/            # performance_report and generated (CSV) reports
strategies/           # indicators & strategies (MA, RSI, Breakout, Bollinger, ATR, PSAR…)
utils/                # settings (paths & keys), chat component, risk mgmt, portfolio, compare_strategies
scripts/              # auto_bot.py (one‑shot/scheduled execution)
data/                 # market data (.csv) — configured by DATA_ROOT
results/              # outputs/figures — configured by RESULT_ROOT
ml/training/          # training datasets — configured by ML_TRAIN_ROOT
ml/trained_models/    # saved models (.json/.pkl) — configured by ML_MODELS_ROOT
logs/                 # logs (e.g., logs/auto_bot_py.log)
```

> Root folders like `DATA_ROOT`, `REPORT_ROOT`, `RESULT_ROOT`, `ML_TRAIN_ROOT`, `ML_MODELS_ROOT` come from **`utils/settings.py`**. Ensure they exist or are created at startup.

---

## ✨ Key features

- **dashboard.py**
  - Yahoo Finance with Alpha Vantage *fallback*; CSV saved under `DATA_ROOT`.
  - Multi‑strategy application (MA, RSI, Breakout, Bollinger, ATR, PSAR) → **reports** (returns/equity/signals).
  - **Strategy comparison** (30‑day Sharpe, MDD, Win Rate, Profit Factor, Score + verdict).
  - Interactive Plotly charts (candles, volume, indicators, equity, buy/sell markers).
  - Chat component (OpenAI): `utils.chat_component.init_chat_with_emilio()`.

- **1_Robots.py**
  - **ML**: feature selection, training (RandomForest / SVC), model save/load, **Top‑N** predictions.
  - **Risk clustering**: select one symbol per cluster + visualization.
  - **Portfolio optimization** across strategies: optimal weights, risk contributions; memo of best combos.

- **2_Alpaca.py**
  - **Account & Portfolio (paper)**, equity, **Risk Dashboard** (annualized vol, VaR/ES 95%).
  - **Order plan** with safeguards (e.g., block if 30‑day Sharpe < 0, cap by k% × VaR), list & **submit** via Alpaca.
  - **One‑shot auto‑bot** via env vars (script `scripts/auto_bot.py`).

---

## 🔧 Prereqs & setup

- Python **3.10+**
- API keys: Alpha Vantage, **Alpaca (paper)**, optional OpenAI & Slack
- Install:
  ```bash
  pip install -U pip
  pip install streamlit yfinance alpha_vantage pandas numpy scipy matplotlib plotly ta scikit-learn               alpaca-trade-api alpaca-py openai streamlit-plotly-events
  ```

### 🔑 Secrets / configuration

Create `.streamlit/secrets.toml`:

```toml
# Market data
ALPHA_API_KEY = "..."

# Alpaca (paper)
ALPACA_API_KEY    = "..."
ALPACA_SECRET_KEY = "..."
ALPACA_PAPER_URL  = "https://paper-api.alpaca.markets"

# Chat (optional)
OPENAI_API_KEY = "..."

# Slack (optional)
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."
```

> Root directories (DATA/REPORT/RESULT/ML) are defined in `utils.settings`.

---

## ▶️ Run (with the CURRENT layout)

Your 3 pages are at the **repository root**. Run them **separately**:

```bash
# Main dashboard
streamlit run dashboard.py

# Robots / ML
streamlit run 1_Robots.py

# Alpaca (account, risk, orders)
streamlit run 2_Alpaca.py
```

### 💡 Multi‑page option (built‑in navigation)
If you prefer Streamlit’s native multipage navigation, place the other scripts under **`pages/`** next to `dashboard.py`:
```
.
├── dashboard.py
└── pages/
    ├── 1_Robots.py
    └── 2_Alpaca.py
```
Then run `streamlit run dashboard.py`.

---

## 🔁 Typical workflow

1. **dashboard.py**: load data → apply strategies → create/compare **report** → interactive chart.  
2. **1_Robots.py**: train/load model → **Top‑N** picks → risk clustering → **optimization** → memo best combos (session).  
3. **2_Alpaca.py**: check account/risk (VaR/ES) → **order plan** (safeguards) → **submit** (paper).

> The **memo** (`strategies_to_execute` in session state) bridges Robots → Alpaca without re‑entry.

---

## ⚠️ Good practices

- Validate in **paper** before any real trading.
- Safeguards (30‑day Sharpe, k% × VaR) **do not eliminate** risk.
- **Never** commit API keys. Use `secrets.toml` or environment variables.

---

## 📄 License

Add your preferred license (e.g., MIT) in `LICENSE`.
