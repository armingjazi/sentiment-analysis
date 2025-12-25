import argparse

from pandas import read_csv

from trading_sentiment_analysis.label.download_sentiment_data import download_financial_news_data
from trading_sentiment_analysis.label.stock_price_label import label_news_data
from trading_sentiment_analysis.stock_api.twelve_data import TwelveData

import os
from dotenv import load_dotenv

from trading_sentiment_analysis.stock_api.cache import CacheData
from trading_sentiment_analysis.stock_api.stock_api import StockAPI


def batch_label_news_data(input_data, output_dir, stock_api: StockAPI, batch_start=0):
    total = len(input_data)
    batch_size = 1000

    for start_idx in range(0, total, batch_size):
        batch_number = start_idx // batch_size
        if batch_number < batch_start:
            continue
        print(f"Processing batch {batch_number}/{(total + batch_size - 1) // batch_size}")
        output_path = os.path.join(output_dir, f'news_batch_{batch_number}.csv')
        end_idx = min(start_idx + batch_size, total)
        batch = input_data.iloc[start_idx:end_idx]

        labeled_batch = label_news_data(batch, stock_api, start_idx)

        labeled_batch.to_csv(output_path)

        print(f"Processed batch {batch_number}")


def main():
    parser = argparse.ArgumentParser(description='Labels news batch by batch')
    parser.add_argument('batch_start', type=int, nargs='?', default=0)
    args = parser.parse_args()
    batch_start = args.batch_start

    load_dotenv()
    cache = CacheData(os.path.join('data', 'stock'))
    twelve_data = TwelveData(os.getenv('TWELVE_DATA_API_KEY'), cache)

    print("labeling news...")

    path = download_financial_news_data()
    input_data = read_csv(os.path.join(path, 'analyst_ratings_processed.csv'))

    output_dir = os.path.dirname(os.path.join('data', 'labeled', 'news.csv'))
    if not os.path.exists(output_dir):
        raise Exception('Output dir does not exist', output_dir)

    batch_label_news_data(input_data, output_dir, twelve_data, batch_start)


if __name__ == '__main__':
    main()