from typing import Protocol

import pandas as pd


class StockAPI(Protocol):
    def download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        pass