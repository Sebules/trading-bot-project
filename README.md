# Trading Bot Dashboard (Streamlit + Alpaca, Risk & Auto-bot)

App Streamlit pour:
- 🧮 **Backtests multi-stratégies** (MA, RSI, Breakout, BB)
- 📈 **Sharpe 30 jours** + remplacement auto si < 0
- 🛡️ **Risk dashboard**: volatilité ex-ante, VaR(95%), ES(95%), limite k%×VaR
- 🧠 **Optimisation de portefeuille** (poids) → mémo → exécution pondérée
- 🤖 **Bot auto** (script) avec garde-fous (k%×VaR, cash)

## Démo (screenshots)
*(insère ici 2–3 captures “Passage d’ordres”, “Bot”, “Risk”)*
<img width="1363" height="742" alt="image" src="https://github.com/user-attachments/assets/25c15f07-c4c9-4f62-b0f9-a2bc18dc19c6" />

<img width="1375" height="737" alt="image" src="https://github.com/user-attachments/assets/fff223bc-dd39-4cc4-b3cb-eb2ad20cc851" />

<img width="725" height="747" alt="image" src="https://github.com/user-attachments/assets/c80c365f-ca9a-4e26-b692-a739172380d7" />

<img width="1847" height="786" alt="image" src="https://github.com/user-attachments/assets/241d6e6a-ab11-4ef3-9bc7-18854c81a115" />




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
