import pandas as pd

def moving_average_crossover_strategy(df, short_window=20, long_window=50):
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    df['Signal'] = 0
    df.loc[df['SMA_short'] > df['SMA_long'], 'Signal'] = 1
    df.loc[df['SMA_short'] < df['SMA_long'], 'Signal'] = -1
    df['Position'] = df['Signal'].diff()
    return df