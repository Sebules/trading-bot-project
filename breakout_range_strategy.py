import pandas as pd


def breakout_range(df):
    # Calcul du range (basé sur les 20 derniers jours)
    lookback = 20
    df['High_Range'] = df['High'].rolling(window=lookback).max()
    df['Low_Range'] = df['Low'].rolling(window=lookback).min()

    # Détection du breakout haussier / baissier
    df['Breakout_Haut'] = df['Close'] > df['High_Range'].shift(1)
    df['Breakout_Bas'] = df['Close'] < df['Low_Range'].shift(1)
    return print(df[['Close', 'High_Range', 'Low_Range', 'Breakout_Haut', 'Breakout_Bas']].tail())