import pandas as pd
from .base import Strategy

class RSIStrategy(Strategy):
    def __init__(self, df, window, rsi_low, rsi_high):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df doit être un DataFrame")
        if "Close" not in df.columns:
            raise ValueError("Le DataFrame doit contenir une colonne 'Close'")

        super().__init__(df)
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        self.window = int(window)
        self.rsi_low = int(rsi_low)
        self.rsi_high = int(rsi_high)


    #CAlculer le RSI sur une periode de 14.
    def compute_rsi(self, series):
        delta = series.diff() # calcul du delta

        #Séparation gain et perte
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        #calcul des moyennes sur la periode choisie
        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()

        #calcul du RS et RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


    # Application de la stratégie RSI à un dataframe donné
    def generate_signals(self):
     self.df = self.df.copy()
     self.df['RSI'] = self.compute_rsi(self.df['Close'])  # création de la colonne RSI

     self.df['Signal'] = 0  # Création de la colonne Signal avec en initial des zéros
     self.df.loc[self.df['RSI'] < self.rsi_low, 'Signal'] = 1  # allouer la valeur de 1 dans la colonne Signal si le rsi est bas (<30)
     self.df.loc[self.df['RSI'] > self.rsi_high, 'Signal'] = -1  # allouer la valeur de -1 dans la colonne Signal si le rsi est haut (>70)

     self.df['Position'] = self.df['Signal'].shift(1)  # créer la colonne position en faisant un décalage de 1 de la colonne Signal
     self.df['Returns'] = self.df['Close'].pct_change()  # Créer une colonne pour le return, .pct_change() permet ce calcul rapidement
     self.df['Strategy'] = self.df['Returns'] * self.df['Position']  # créer la colonne strategy

     return self.df