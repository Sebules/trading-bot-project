import pandas as pd
#CAlculer le RSI sur une periode de 14.
def compute_rsi(series, window=14):
    delta = series.diff() # calcul du delta

    #Séparation gain et perte
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    #calcul des moyennes sur la periode choisie
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    #calcul du RS et RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Application de la stratégie RSI à un dataframe donné
def rsi_strategy(df, rsi_low=30, rsi_high=70):
 df = df.copy()
 df['RSI'] = compute_rsi(df['Close'])  # création de la colonne RSI

 df['Signal'] = 0  # Création de la colonne Signal avec en initial des zéros
 df.loc[df['RSI'] < rsi_low, 'Signal'] = 1  # allouer la valeur de 1 dans la colonne Signal si le rsi est bas (<30)
 df.loc[df['RSI'] > rsi_high, 'Signal'] = -1  # allouer la valeur de -1 dans la colonne Signal si le rsi est haut (>70)

 df['Position'] = df['Signal'].shift(1)  # créer la colonne position en faisant un décalage de 1 de la colonne Signal
 df['Returns'] = df['Close'].pct_change()  # Créer une colonne pour le return, .pct_change() permet ce calcul rapidement
 df['Strategy'] = df['Returns'] * df['Position']  # créer la colonne strategy

 return df