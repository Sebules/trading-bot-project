# 📊 Trading Algo Dashboard (Streamlit)

Un tableau de bord **multi-outils** pour l’analyse de marché, la comparaison de stratégies, le ML (Machine Learning), le risk management et l’**exécution (paper)** via Alpaca.

> ⚠️ **Avertissement** — Ce projet est fourni à titre éducatif. Rien n’est un conseil financier. Testez en *paper trading*.

---

## 🗂️ Arborescence du dépôt

```
.
└── streamlit_app
  ├── dashboard.py      # Tableau de bord principal (chargement données, stratégies, comparaisons, graphes)
  └── pages/
      ├── 1_Robots.py  # Robots: ML (RF/SVC), clustering risque, optimisation de portefeuille
      └── 2_Alpaca.py  # Intégration Alpaca: compte/positions, Risk Dashboard (VaR/ES), passage d'ordres
└── (répertoires Python requis par les imports, voir ci-dessous)
```

Le code importe des modules internes attendus dans les dossiers suivants (présents dans ton dépôt, même s’ils ne sont pas joints ici) :

```
broker/               # cash, exécution Alpaca, règles, persistance, métriques, etc.
execution/            # exécution de signaux, run_bot, optimiseur de stratégies
ml/                   # chargement/entraînement modèles, pickles & métadonnées
reporting/            # performance_report et (CSV) de rapports générés
strategies/           # indicateurs & stratégies (MA, RSI, Breakout, Bollinger, ATR, PSAR…)
utils/                # settings (chemins & clés), chat component, risk mgmt, portfolio, compare_strategies
scripts/              # auto_bot.py (exécution one‑shot/planifiée)
data/                 # données marché (.csv) — défini par DATA_ROOT
results/              # sorties/figures — défini par RESULT_ROOT
ml/training/          # jeux d'entraînement — défini par ML_TRAIN_ROOT
ml/trained_models/    # modèles sauvegardés (.json/.pkl) — défini par ML_MODELS_ROOT
logs/                 # journaux (ex. logs/auto_bot_py.log)
```

> Les chemins `DATA_ROOT`, `REPORT_ROOT`, `RESULT_ROOT`, `ML_TRAIN_ROOT`, `ML_MODELS_ROOT` sont **centralisés dans `utils/settings.py`**. Assure‑toi que ces dossiers existent ou qu’ils sont créés au démarrage.

---

## ✨ Fonctionnalités principales

- **dashboard.py**
  - Chargement via Yahoo Finance avec *fallback* Alpha Vantage; sauvegarde en CSV dans `DATA_ROOT`.
  - Application multi‑stratégies (MA, RSI, Breakout, Bollinger, ATR, PSAR), génération de **rapports** (returns/equity/signaux).
  - **Comparaison de stratégies** (Sharpe 30j, MDD, Win Rate, Profit Factor, Score + verdict).
  - Graphiques Plotly interactifs (chandeliers, volume, indicateurs, equity, marqueurs buy/sell).
  - Composant de chat (OpenAI) : `utils.chat_component.init_chat_with_emilio()`.

- **1_Robots.py**
  - **ML** : sélection de features, entraînement (RandomForest / SVC), export & rechargement de modèles, prédiction Top‑N.
  - **Clustering de risque** : sélection d’un titre par cluster + visualisation.
  - **Optimisation de portefeuille multi‑stratégies** : poids optimaux, contributions au risque; mémo des meilleures combinaisons.

- **2_Alpaca.py**
  - **Compte & Portefeuille (paper)**, equity, **Risk Dashboard** (Vol annualisée, VaR/ES 95 %).
  - **Plan d’ordres** avec garde‑fous (p.ex. blocage si Sharpe 30j < 0, plafond k % × VaR), liste et **soumission** via Alpaca.
  - **Auto‑bot one‑shot** : exécution locale via variables d’environnement (script `scripts/auto_bot.py`).

<img width="725" height="747" alt="image_ALPACA_temps_reel" src="https://github.com/user-attachments/assets/6bcb9f20-14e3-4fc5-8800-dea15a4bdbfe" />
<img width="1332" height="618" alt="image_ALPACA_temps_reel-diagramme" src="https://github.com/user-attachments/assets/5c476902-2e68-4559-ac7a-dd04a16846e4" />
<img width="1386" height="747" alt="image_ALPACA_temps_reel-risk_dashboard" src="https://github.com/user-attachments/assets/213955fd-991d-475a-b0b7-1233aee2a84c" />
<img width="1332" height="848" alt="image_ALPACA_temps_reel-liste_ordres" src="https://github.com/user-attachments/assets/8a255fb1-10f2-4907-a2f7-52acfe4a03e8" />
<img width="1363" height="742" alt="image_ALPACA_passage_ordres" src="https://github.com/user-attachments/assets/1453f763-b782-4c26-9b55-d5c34040cfcf" />
<img width="1375" height="737" alt="image_ALPACA_plan_reequilibrage" src="https://github.com/user-attachments/assets/8a1bb6ce-9e80-4fe4-83c3-90d808b5f525" />
<img width="1391" height="460" alt="image_ALPACA_autobot" src="https://github.com/user-attachments/assets/93c7fc16-9e3f-4aad-a215-f312dcf8a5f0" />

---

## 🔧 Prérequis & installation

- Python **3.10+**
- Clés/API : Alpha Vantage, **Alpaca (paper)**, (optionnel) OpenAI & Slack
- Installation (recommandée) :
  ```bash
  pip install -U pip
  pip install streamlit yfinance alpha_vantage pandas numpy scipy matplotlib plotly ta scikit-learn               alpaca-trade-api alpaca-py openai streamlit-plotly-events
  ```

### 🔑 Secrets / configuration

Créer `.streamlit/secrets.toml` dans le dossier **`streamlit_app/`** ou rajouter les secrets dans le le fichier `.env`:

```toml
# Market data
ALPHA_API_KEY = "..."

# Alpaca (paper)
ALPACA_API_KEY    = "..."
ALPACA_SECRET_KEY = "..."
ALPACA_PAPER_URL  = "https://paper-api.alpaca.markets"

# Chat (optionnel)
OPENAI_API_KEY = "..."

# Slack (optionnel)
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."
```

> Les répertoires racine (DATA/REPORT/RESULT/ML) proviennent de `utils.settings`.

---

## ▶️ Lancer l’application

Tes 3 pages sont dans le dossier **`streamlit_app/`** du dépôt. `1_Robots.py` et `2_Alpaca.py` sont dans le dossier **`pages/`** à côté de `dashboard.py`. Lance l'appli multipage avec menu :

```bash
streamlit run dashboard.py
```

## 🔁 Flux typique

1. **dashboard.py** : charger données → appliquer stratégies → générer & comparer un **rapport** → graphe interactif.  
2. **1_Robots.py** : entraîner/charger modèle → **Top‑N** tickers → clustering risque → **optimisation** → mémo des meilleures combinaisons (session).  
3. **2_Alpaca.py** : vérifier compte/risque (VaR/ES) → **plan d’ordres** (garde‑fous) → **soumission** (paper).

> Le **mémo** de stratégies (session `strategies_to_execute`) permet de passer de Robots → Alpaca sans ressaisie.

---

## ⚠️ Bonnes pratiques

- Toujours valider en **paper** avant toute exécution réelle.
- Les garde‑fous (Sharpe 30j, k % × VaR) ne suppriment pas le risque.
- Ne **jamais** committer les clés/API. Utiliser `secrets.toml` ou des variables d’environnement.

---
__________________________________________________________________________________________________________________________________

# 📊 Trading Algo Dashboard (Streamlit)

A **multi‑tool** dashboard for market analysis, strategy comparison, ML, risk management, and **(paper) execution** via Alpaca.

> ⚠️ **Disclaimer** — For educational purposes only. This is *not* financial advice. Test in **paper trading** first.

---

## 🗂️ repository layout

```
.
└── streamlit_app
  ├── dashboard.py      # Main dashboard (data loading, strategies, comparisons, charts)
  └── pages/
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

## ▶️ Run

Your 3 pages are at the **`streamlit_app`**. `1_Robots.py` et `2_Alpaca.py` are under **`pages/`** netx to `dashboard.py`. Run Streamlit’s native multipage navigation :

```bash
streamlit run dashboard.py
```
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




