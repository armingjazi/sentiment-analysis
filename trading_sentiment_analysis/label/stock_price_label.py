import numpy as np
import pandas as pd
from datetime import timedelta

from trading_sentiment_analysis.stock_api.stock_api import StockAPI
from trading_sentiment_analysis.stock_api.twelve_data import CachedSymbolFailureException

# Track which cached failures we've already logged to reduce noise
_logged_cached_failures = set()


def get_stock_movement(ticker, news_date, stock_api: StockAPI, window=1):
    """
    Get stock price movement after news release

    Parameters:
    ticker (str): Stock ticker symbol
    news_date (str): Date of news in format YYYY-MM-DD HH:MM:SS
    window (int): Number of trading days to look ahead
    stock_api (yfinance.Ticker): Stock API object

    Returns:
    float: Percentage price change
    """
    try:
        # Convert news_date to datetime

        ## check if date is timezone aware, if not convert to GMT+4
        if pd.to_datetime(news_date).tzinfo is None:
            news_dt = pd.to_datetime(news_date).tz_localize('Etc/GMT+4')
        else:
            news_dt = pd.to_datetime(news_date)

        # Get end date (adding a few extra days to ensure we get next trading day)
        end_dt = news_dt + timedelta(days=window + 20)
        start_dt = news_dt - timedelta(days=20)

        # Download stock data
        stock = stock_api.download(
            ticker,
            start=start_dt,
            end=end_dt,
        )

        if stock.index.tz is None:
            stock = stock.tz_localize('Etc/GMT+4')

        if len(stock) < 2:
            return np.nan

        # Get the closest prices before and after news
        try:
            prices_before = stock['Close'].loc[:news_dt]
            prices_after = stock['Close'].loc[news_dt:]

            if len(prices_before) == 0 or len(prices_after) == 0:
                return np.nan

            price_before = float(prices_before.iloc[-1])
            price_after = float(prices_after.iloc[0])
        except Exception as e:
            print(f"Error processing {ticker} for date {news_date}, {str(e)}")
            print(f"stock {stock}")
            print(f"news_dt {news_dt}")
            return np.nan


        # Calculate percentage change
        return ((price_after - price_before) / price_before) * 100

    except CachedSymbolFailureException as e:
        # Only log each cached failure once per run
        if ticker not in _logged_cached_failures:
            print(f"Skipping {ticker}: previously returned 404 (will skip silently for remaining occurrences)")
            _logged_cached_failures.add(ticker)
        return np.nan
    except Exception as e:
        print(f"Error processing {ticker} for date {news_date}: {str(e)}")
        return np.nan


def label_news_data(news_df, stock_api: StockAPI, start_idx, threshold=1.0):
    """
    Label news based on subsequent stock price movement

    Parameters:
    news_df (pd.DataFrame): DataFrame with columns 'stock', 'date', 'title'
    threshold (float): Percentage threshold for considering movement significant
    stock_api (yfinance.Ticker): Stock API object

    Returns:
    pd.DataFrame: Original dataframe with added 'label' and 'price_change' columns
    """
    # Copy dataframe
    df = news_df.copy()

    # Initialize new columns
    df['price_change'] = np.nan
    df['label'] = np.nan

    # Process each news item
    for idx in range(len(df)):
        batch_idx = start_idx + idx

        # Get price movement
        stock_val = df.at[batch_idx, 'stock']
        date_val = df.at[batch_idx, 'date']

        # Check for data corruption
        if pd.isna(stock_val) or pd.isna(date_val):
            continue

        change = get_stock_movement(stock_val, date_val, stock_api)

        # Store price change
        df.at[batch_idx, 'price_change'] = change

        # Label based on threshold
        if pd.notna(change):
            if change > threshold:
                df.at[batch_idx, 'label'] = 1  # Positive
            elif change < -threshold:
                df.at[batch_idx, 'label'] = 0  # Negative
            else:
                df.at[batch_idx, 'label'] = 2  # Neutral
    return df
