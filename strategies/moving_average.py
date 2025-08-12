import pandas as pd
from .base import Strategy #mettre un point devant base, sinon l'import dans le dashboard ne fonctionne pas

class MovingAverageStrategy(Strategy):
    def __init__(self, df, short_window, long_window):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df doit être un DataFrame")
        if "Close" not in df.columns:
            raise ValueError("Le DataFrame doit contenir une colonne 'Close'")

        super().__init__(df)
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        self.short_window = int(short_window)
        self.long_window = int(long_window)

    def generate_signals(self):
        # Calcul des moyennes mobiles
        self.df["SMA_Short"] = self.df["Close"].rolling(window=self.short_window).mean()
        self.df["SMA_Long"] = self.df["Close"].rolling(window=self.long_window).mean()

        # Initialisation des signaux
        self.df["Signal"] = 0
        self.df.loc[self.df.index[self.short_window:], "Signal"] = (
            self.df["SMA_Short"].iloc[self.short_window:] > self.df["SMA_Long"].iloc[self.short_window:]
        ).astype(int)

        self.df["Equity_Curve"] = (1 + self.df["Signal"].shift(1) * self.df["Close"].pct_change()).cumprod()

        return self.df