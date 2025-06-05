import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from reporting.performance_report import generate_verdict

# Chargement des résultats (à adapter)
df = pd.read_csv("C:/Users/sebas/Documents/trading_bot_project/backtest_results_RandomForest_ML.csv")
report = pd.read_csv("C:/Users/sebas/Documents/trading_bot_project/reporting/rapport_RandomForest ML.csv", index_col=0).squeeze()

st.title("Rapport de Stratégie AlgoTrading")

st.subheader("Courbe des rendements cumulés")
fig, ax = plt.subplots()
ax.plot(df["Equity_Curve"], label="Stratégie")
ax.set_ylabel("Évolution du capital")
ax.set_xlabel("Temps")
ax.legend()
st.pyplot(fig)

st.subheader("Statistiques de performance")
st.dataframe(report)

decision, reasons = generate_verdict(report)
st.success(f"**Verdict final :** {decision}")
st.markdown(f"**Motifs :** {', '.join(reasons)}")