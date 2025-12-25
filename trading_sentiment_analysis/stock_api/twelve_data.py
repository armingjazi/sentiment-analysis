import time

import pandas as pd
import requests
from trading_sentiment_analysis.stock_api.cache import CacheData
from typing import Callable
from functools import wraps

class RateLimitException(Exception):
    def __init__(self, message="API rate limit exceeded", retry_after=60):
        self.retry_after = retry_after
        super().__init__(message)

class CachedSymbolFailureException(Exception):
    def __init__(self, symbol):
        self.symbol = symbol
        super().__init__(f'Symbol {symbol} previously returned 404 (cached failure)')

def retry_on_rate_limit(long_wait=86400,max_retries_for_long_wait: int =3, max_retries: int = 999999):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except RateLimitException as e:
                    retries += 1

                    if retries == max_retries_for_long_wait:
                        print(f"Long Rate limit hit. Waiting {long_wait} seconds...")
                        time.sleep(long_wait)
                    elif retries > max_retries:
                        raise e
                    print(f"Rate limit hit. Waiting {e.retry_after} seconds...")
                    time.sleep(e.retry_after)
            return func(*args, **kwargs)
        return wrapper
    return decorator


class TwelveData:
    def __init__(self, api_key, cache_data:CacheData=None):
        self.api_key = api_key
        self.base_url = 'https://api.twelvedata.com'
        self.headers = {
            'Content-Type': 'application/json'
        }
        self.cache_data = cache_data

    @retry_on_rate_limit()
    def api_request(self, url):
        response = requests.get(url, headers=self.headers)
        data = response.json()
        # status code check
        if response.status_code != 200:
            print(f"Failed API request: {data}")
            raise ValueError(f'API request failed with status code {response.status_code}')
        if 'code' in data and data['code'] == 429:
            print(f"Rate limit exceeded: {data}")
            raise RateLimitException(retry_after=60)
        if 'code' in data and data['code'] == 404:
            print(f"Symbol not found: {data}")
            raise ValueError(f'API request failed with error code 404, symbol not found')
        if 'code' in data and data['code'] != 200:
            print(f"API request error: {data}")
            raise ValueError(f'API request failed with error code {data.code}')
        
        return data

    def cache_or_download(self, symbol):
        from_cache = False
        # check cache
        # if cache exists, load from cache
        if self.cache_data is not None:
            data = self.cache_data.get_data(symbol)
            if data is not None:
                from_cache = True
                return data, from_cache

            # Check if this symbol previously returned 404
            if self.cache_data.is_failed_symbol(symbol):
                raise CachedSymbolFailureException(symbol)

        url = f'{self.base_url}/time_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={self.api_key}&timezone=utc'

        try:
            data = self.api_request(url)
        except ValueError as e:
            # Check if this is a 404 error
            if '404' in str(e) and 'symbol not found' in str(e):
                # Mark this symbol as failed to prevent future API calls
                if self.cache_data is not None:
                    self.cache_data.mark_symbol_as_failed(symbol)
            # Re-raise the exception
            raise

        df = pd.DataFrame(data['values'])
        df.index = pd.DatetimeIndex(pd.to_datetime(df['datetime']))
        df = df.drop(columns=['datetime'])
        df.rename(columns={'close': 'Close'}, inplace=True)
        df.rename(columns={'open': 'Open'}, inplace=True)
        df.rename(columns={'high': 'High'}, inplace=True)
        df.rename(columns={'low': 'Low'}, inplace=True)
        df.rename(columns={'volume': 'Volume'}, inplace=True)

        return df, from_cache

    def download(self, symbol, start, end):
        df, from_cache = self.cache_or_download(symbol)

        # write to csv file as cache for future use if not empty
        if self.cache_data is not None and not from_cache:
            if not df.empty:
                self.cache_data.save_data(df, symbol)

        # filter the data based on the start and end dates
        df = df[(df.index >= pd.to_datetime(start).asm8) & (df.index <= pd.to_datetime(end).asm8)]

        return df
