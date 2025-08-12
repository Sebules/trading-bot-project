class Strategy:
    def __init__(self, df):
        self.df = df.copy()

    def generate_signals(self):
        raise NotImplementedError("Cette méthode doit être implémentée.")