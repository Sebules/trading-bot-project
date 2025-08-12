# Trading Bot Dashboard (Streamlit + Alpaca, Risk & Auto-bot)

App Streamlit pour:
- 🧮 **Backtests multi-stratégies** (MA, RSI, Breakout, BB)
- 📈 **Sharpe 30 jours** + remplacement auto si < 0
- 🛡️ **Risk dashboard**: volatilité ex-ante, VaR(95%), ES(95%), limite k%×VaR
- 🧠 **Optimisation de portefeuille** (poids) → mémo → exécution pondérée
- 🤖 **Bot auto** (script) avec garde-fous (k%×VaR, cash)

## Démo (screenshots)
*(insère ici 2–3 captures “Passage d’ordres”, “Bot”, “Risk”)*

## Arbo rapide
dashboard.py # page principale (Données/Paramétrages/Graphiques)
pages/1_Robots.py # optimisation, ML, mémo
pages/2_Alpaca.py # temps réel, passage d’ordres, bot (UI)
broker/, execution/, ... # modules internes
scripts/auto_bot.py # bot automatique (cron / tâche planifiée)



## Installation locale
```bash
pip install -r requirements.txt
streamlit run dashboard.py