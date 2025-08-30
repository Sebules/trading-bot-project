# 📊 Trading Algo Dashboard (Streamlit)

Un tableau de bord **multi-outils** pour l’analyse de marché, la comparaison de stratégies, le ML, le risk management et l’**exécution (paper)** via Alpaca.

> ⚠️ **Avertissement** — Ce projet est fourni à titre éducatif. Rien n’est un conseil financier. Testez en *paper trading*.

---

## 🗂️ Arborescence ACTUELLE du dépôt

```
.
├── dashboard.py      # Tableau de bord principal (chargement données, stratégies, comparaisons, graphes)
├── 1_Robots.py       # Robots: ML (RF/SVC), clustering risque, optimisation de portefeuille
├── 2_Alpaca.py       # Intégration Alpaca: compte/positions, Risk Dashboard (VaR/ES), passage d'ordres
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

Créer `.streamlit/secrets.toml` :

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

## ▶️ Lancer l’application (avec arborescence ACTUELLE)

Tes 3 pages sont à la **racine** du dépôt. Lance‑les **séparément** :

```bash
# Tableau de bord principal
streamlit run dashboard.py

# Robots / ML
streamlit run 1_Robots.py

# Alpaca (compte, risque, ordres)
streamlit run 2_Alpaca.py
```

### 💡 Option multi‑pages (navigation automatique)
Si tu préfères une appli multipage avec menu, déplace ces fichiers dans un dossier **`pages/`** à côté de `dashboard.py` :
```
.
├── dashboard.py
└── pages/
    ├── 1_Robots.py
    └── 2_Alpaca.py
```
Puis lance : `streamlit run dashboard.py`.

---

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

## 📄 Licence

Ajoute la licence de ton choix (ex. MIT) dans `LICENSE`.
