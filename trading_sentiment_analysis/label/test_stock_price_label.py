import unittest

import pandas as pd

from unittest.mock import Mock
from datetime import datetime

from trading_sentiment_analysis.label.stock_price_label import get_stock_movement, label_news_data

class MockStockAPI:
    def __init__(self, download_ret_value):
        self.download = Mock()
        self.download.return_value = download_ret_value

    def download(self, ticker, start, end):
        pass


class ProcessDataTestCase(unittest.TestCase):
    def test_get_stock_movement(self):
        dates = pd.date_range(
            start='2021-01-01',
            end='2021-01-06',
            freq='D',
            tz='Etc/GMT+4'
        )

        mock_data = {
            'Close': [100, 110, 121, 131, 130, 140],
        }

        stock_mock = MockStockAPI(pd.DataFrame(mock_data, index=dates))

        price_movement = get_stock_movement('AAPL',
                                            datetime.fromisoformat('2021-01-01'),
                                            stock_mock, 1)

        self.assertEqual(0, price_movement)

    def test_get_stock_movement_exact_news_date_does_not_exist(self):
        dates = [
            datetime.fromisoformat('2012-01-03 00:00:00-04:00'),
            datetime.fromisoformat('2011-12-30 00:00:00-04:00'),
            datetime.fromisoformat('2011-12-29 00:00:00-04:00'),
            datetime.fromisoformat('2011-12-28 00:00:00-04:00'),
            datetime.fromisoformat('2011-12-23 00:00:00-04:00'),
        ]

        mock_data = {
            'Close': [100, 110, 121, 131, 120],
        }

        stock_mock = MockStockAPI(pd.DataFrame(mock_data, index=dates))

        price_movement = get_stock_movement('AAPL',
                                            datetime.fromisoformat('2011-12-27'),
                                            stock_mock, 1)

        self.assertAlmostEqual(-8.4, price_movement, 1)

    def test_get_stock_movement_no_data(self):
        dates = pd.date_range(
            start='2021-01-02',
            end='2021-01-06',
            freq='D',
            tz='Etc/GMT+4'
        )

        mock_data = {
            'Close': [100, 110, 121, 131, 130],
        }

        stock_mock = MockStockAPI(pd.DataFrame(mock_data, index=dates))

        price_movement = get_stock_movement('AAPL',
                                            datetime.fromisoformat('2021-01-01'),
                                            stock_mock, 1)

        self.assertTrue(pd.isna(price_movement))

    def test_get_stock_movement_tz_aware(self):
        dates = pd.date_range(
            start='2021-01-02',
            end='2021-01-06',
            freq='D',
            tz='Etc/GMT+4'
        )

        mock_data = {
            'Close': [100, 110, 121, 131, 130],
        }

        stock_mock = MockStockAPI(pd.DataFrame(mock_data, index=dates))

        price_movement = get_stock_movement('AAPL',
                                            '2021-01-03 04:30:00-04:00',
                                            stock_mock, 1)

        self.assertEqual(10, price_movement)

    def test_label_news_data_positive(self):
        news_df = pd.DataFrame({
            'Unnamed: 0': [0.0],
            'title': [
                'News Article 1',
            ],
            'date': [
                '2021-01-04 04:30:00-04:00',
            ],
            'stock': ['AAPL']
        })

        dates = pd.date_range(
            start='2021-01-04',
            end='2021-01-09',
            freq='D',
            tz='Etc/GMT+4'
        )

        mock_data = {
            'Close': [100, 110, 121, 131, 130, 140],
        }

        mock_stock_data = pd.DataFrame(
            mock_data,
            index=dates
        )

        stock_mock = MockStockAPI(pd.DataFrame(mock_stock_data, index=dates))

        labeled_news_df = label_news_data(news_df, stock_mock, 0)

        self.assertEqual([10], labeled_news_df['price_change'].tolist())
        self.assertEqual( [1], labeled_news_df['label'].tolist())


    def test_label_news_data_negative(self):
        news_df = pd.DataFrame({
            'Unnamed: 0': [0.0],
            'title': [
                'News Article 1',
            ],
            'date': [
                '2021-01-04 04:30:00-04:00',
            ],
            'stock': ['AAPL']
        })

        dates = pd.date_range(
            start='2021-01-04',
            end='2021-01-09',
            freq='D',
            tz='Etc/GMT+4'
        )

        mock_data = {
            'Close': [100, 90, 121, 131, 130, 140],
        }

        mock_stock_data = pd.DataFrame(
            mock_data,
            index=dates
        )

        stock_mock = MockStockAPI(pd.DataFrame(mock_stock_data, index=dates))

        labeled_news_df = label_news_data(news_df, stock_mock, 0)

        self.assertEqual([-10], labeled_news_df['price_change'].tolist())
        self.assertEqual( [0], labeled_news_df['label'].tolist())

    def test_label_news_data_neutral(self):
        news_df = pd.DataFrame({
            'Unnamed: 0': [0.0],
            'title': [
                'News Article 1',
            ],
            'date': [
                '2021-01-04 04:30:00-04:00',
            ],
            'stock': ['AAPL']
        })

        dates = pd.date_range(
            start='2021-01-04',
            end='2021-01-09',
            freq='D',
            tz='Etc/GMT+4'
        )

        mock_data = {
            'Close': [100, 101, 121, 131, 130, 140],
        }

        mock_stock_data = pd.DataFrame(
            mock_data,
            index=dates
        )

        stock_mock = MockStockAPI(pd.DataFrame(mock_stock_data, index=dates))

        labeled_news_df = label_news_data(news_df, stock_mock, 0)

        self.assertEqual([1.], labeled_news_df['price_change'].tolist())
        self.assertEqual( [2], labeled_news_df['label'].tolist())

if __name__ == '__main__':
    unittest.main()
