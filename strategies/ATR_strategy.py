import pandas as pd
import ta
from .base import Strategy

class ATRStrategy(Strategy):
    def __init__(self, df, window_ATR):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df doit être un DataFrame")
        if "Close" not in df.columns:
            raise ValueError("Le DataFrame doit contenir une colonne 'Close'")

        super().__init__(df)
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        self.window_ATR = int(window_ATR)

        atr = ta.volatility.AverageTrueRange(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], window=self.window_ATR)
        self.df['ATR'] = atr.average_true_range()

    # Application de la stratégie RSI à un dataframe donné
    def generate_signals(self):

        #For Long Trade
        self.df['Entry_Price'] = self.df['Close'].shift(1)  # Assume buy at the previous candle

        self.df['Stop_Loss_ATR_L'] = self.df['Entry_Price'] - 1.5 * self.df['ATR']
        self.df['Take_Profit_ATR_L'] = self.df['Entry_Price'] + 3 * self.df['ATR']

        #For Short trade
        self.df['Stop_Loss_ATR_S'] = self.df['Entry_Price'] + 1.5 * self.df['ATR']
        self.df['Take_Profit_ATR_S'] = self.df['Entry_Price'] - 3 * self.df['ATR']


        return self.df