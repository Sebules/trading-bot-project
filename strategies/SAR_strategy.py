import pandas as pd
import ta
from .base import Strategy

class PSARStrategy(Strategy):
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df doit être un DataFrame")
        if "Close" not in df.columns:
            raise ValueError("Le DataFrame doit contenir une colonne 'Close'")

        super().__init__(df)
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        psar = ta.trend.PSARIndicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'])
        self.df['PSAR'] = psar.psar()

    # Application de la stratégie RSI à un dataframe donné
    def generate_signals(self):
        self.df['Stop_Loss'] = self.df['PSAR']
        return self.df