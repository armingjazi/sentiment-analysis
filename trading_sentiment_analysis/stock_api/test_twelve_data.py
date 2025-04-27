import unittest

import os
from dotenv import load_dotenv
from pandas import Timestamp

load_dotenv()

from trading_sentiment_analysis.stock_api.twelve_data import TwelveData


class TwelveDataTestCase(unittest.TestCase):

    def test_download(self):
        twelve_data = TwelveData(os.getenv('TWELVE_DATA_API_KEY'))

        df = twelve_data.download('AAPL', Timestamp('2021-01-01'), Timestamp('2021-01-05'))

        self.assertEqual((2, 5), df.shape)
        self.assertEqual(Timestamp('2021-01-05'), df.index[0])
        self.assertEqual(Timestamp('2021-01-04'), df.index[1])