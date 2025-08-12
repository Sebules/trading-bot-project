import pandas as pd
import ta
from ta.volatility import BollingerBands
from .base import Strategy #mettre un point devant base, sinon l'import dans le dashboard ne fonctionne pas

class BBStrategy(Strategy):
    def __init__(self, df, window_bb, window_dev_bb):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df doit être un DataFrame")
        if "Close" not in df.columns:
            raise ValueError("Le DataFrame doit contenir une colonne 'Close'")

        super().__init__(df)
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        self.window_bb = int(window_bb)
        self.window_dev_bb = int(window_dev_bb)



    # Application de la stratégie BB à un dataframe donné
    def generate_signals(self):

        bb = BollingerBands(close=self.df["Close"], window=self.window_bb, window_dev=self.window_dev_bb)

        self.df["bb_middle"] = bb.bollinger_mavg()
        self.df["bb_upper"] = bb.bollinger_hband()
        self.df["bb_lower"] = bb.bollinger_lband()


        self.df["Signal"] = 0  # Création de la colonne Signal avec en initial des zéros
        self.df.loc[self.df["Close"] < self.df["bb_lower"], "Signal"] = 1  # allouer la valeur de 1 dans la colonne Signal si le bb est bas
        self.df.loc[self.df["Close"] > self.df["bb_upper"], "Signal"] = -1  # allouer la valeur de -1 dans la colonne Signal si le bb est haut

        self.df['Position'] = self.df['Signal'].shift(1)  # créer la colonne position en faisant un décalage de 1 de la colonne Signal
        self.df['Returns'] = self.df['Close'].pct_change()  # Créer une colonne pour le return, .pct_change() permet ce calcul rapidement
        self.df['Strategy'] = self.df['Returns'] * self.df['Position']  # créer la colonne strategy

        return self.df