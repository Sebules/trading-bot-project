import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from strategies.moving_average_crossover import moving_average_crossover_strategy

df = pd.read_csv('C:/Users/sebas/Documents/TRADING-BOT-PROJECT/data/btc_usdt.csv')
df = moving_average_crossover_strategy(df)
print(df.head())

# Visualiser
plt.figure(figsize=(14,7))
plt.plot(df['Close'], label='Price')
plt.plot(df['SMA_short'], label='SMA short')
plt.plot(df['SMA_long'], label='SMA long')
plt.legend()
plt.show()

# Performance basique
df['Returns'] = df['Close'].pct_change() #calcul du rendement pour chaque ligne en pourcentage. .pct_change() vient prendre la valeur de la ligne,
#la soustraire à la valeur de la ligne précédente et la diviser par 100.
df['Strategy'] = df['Returns'] * df['Position'].shift(1)
cumulative = (1 + df[['Returns', 'Strategy']]).cumprod()

cumulative.plot(figsize=(14,7), title="Buy & Hold vs Strategy")
plt.show()

AAPL_df = pd.read_csv('C:/Users/sebas/Documents/TRADING-BOT-PROJECT/data/AAPL-11042025.csv')
AAPL_df = moving_average_crossover_strategy(AAPL_df)
print(AAPL_df.head())
# Visualiser
plt.figure(figsize=(14,7))
plt.plot(AAPL_df['Close'], label='Price')
plt.plot(AAPL_df['SMA_short'], label='SMA short')
plt.plot(AAPL_df['SMA_long'], label='SMA long')
plt.legend()
plt.show()
#print(AAPL_df.info())
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.offline as py
# For Notebooks
#init_notebook_mode(connected=True)
# For offline use
cf.go_offline()
AAPL_df[['Close','SMA_short','SMA_long']].iplot(validated=False, title="AAPL Price & Moving Averages")

#import plotly.graph_objects as go

#fig = go.Figure()

#fig.add_trace(go.Scatter(y=AAPL_df['Close'], name='Close'))
#fig.add_trace(go.Scatter(y=AAPL_df['SMA_short'], name='SMA_short'))
#fig.add_trace(go.Scatter(y=AAPL_df['SMA_long'], name='SMA_long'))

#fig.update_layout(title="AAPL Price & Moving Averages",
#                  xaxis_title="Date",
#                  yaxis_title="Price",
#                  template="plotly_dark")

#fig.show()