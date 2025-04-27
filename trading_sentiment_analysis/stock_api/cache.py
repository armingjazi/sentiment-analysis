import os
from typing import Protocol

import pandas as pd


class CacheData(Protocol):
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def get_data(self, ticker) -> pd.DataFrame | None:
        if (self.cache_dir is not None) and os.path.exists(self.cache_dir):
            path = os.path.join(self.cache_dir, f"{ticker}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index)
                return df
        else:
            return None

    def save_data(self, df, ticker):
        if (self.cache_dir is not None) and os.path.exists(self.cache_dir):
            path = os.path.join(self.cache_dir, f"{ticker}.csv")
            df.to_csv(path)
        return
