import pandas as pd
from .base import Strategy

class BreakoutRangeStrategy(Strategy):
    def __init__(self, df, lookback):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df doit être un DataFrame")
        if "Close" not in df.columns:
            raise ValueError("Le DataFrame doit contenir une colonne 'Close'")

        super().__init__(df)
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        self.lookback = int(lookback)


    def generate_signals(self):


        self.df['High_Range'] = self.df['High'].rolling(window=self.lookback).max()
        self.df['Low_Range'] = self.df['Low'].rolling(window=self.lookback).min()

        # Détection du breakout haussier / baissier
        self.df['Breakout_Haut'] = self.df['Close'] > self.df['High_Range'].shift(1)
        self.df['Breakout_Bas'] = self.df['Close'] < self.df['Low_Range'].shift(1)

        # Ajout de la colonne Signal
        self.df['Signal'] = 0
        self.df.loc[self.df['Breakout_Haut'], 'Signal'] = 1
        self.df.loc[self.df['Breakout_Bas'], 'Signal'] = -1

        # Position décalée (comme dans RSI)
        self.df['Position'] = self.df['Signal'].shift(1)

        # Retour & Equity Curve
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Strategy'] = self.df['Returns'] * self.df['Position']
        self.df['Breakout_Equity'] = (1 + self.df['Strategy'].fillna(0)).cumprod()
        return self.df #print(self.df[['Close', 'High_Range', 'Low_Range', 'Breakout_Haut', 'Breakout_Bas']].tail())