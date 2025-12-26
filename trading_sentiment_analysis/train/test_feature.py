import unittest

from trading_sentiment_analysis.train.feature import extract_features


class FeatureTestCase(unittest.TestCase):
    def test_feature_extraction(self):
        features = extract_features('This is a test', frequencies={
            ('This', 1): 1,
            ('is', 1): 1,
            ('a', 1): 1,
            ('test', 1): 2,
        })
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].shape, (3,))
        self.assertEqual(features[0][0], 1)
        self.assertEqual(features[0][1], 2)
        self.assertEqual(features[0][2], 0)

    def test_feature_extraction_empty(self):
        features = extract_features('', frequencies={
            ('This', 1): 1,
            ('is', 1): 1,
            ('a', 1): 1,
            ('test', 1): 2,
        })
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].shape, (3,))
        self.assertEqual(features[0][0], 1)
        self.assertEqual(features[0][1], 0)
        self.assertEqual(features[0][2], 0)

    def test_feature_extraction_no_frequencies(self):
        features = extract_features('This is a test', frequencies={})
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].shape, (3,))
        self.assertEqual(features[0][0], 1)
        self.assertEqual(features[0][1], 0)
        self.assertEqual(features[0][2], 0)

    def test_feature_extraction_no_words(self):
        features = extract_features('no words', frequencies={
            ('This', 1): 1,
            ('is', 1): 1,
            ('a', 1): 1,
            ('test', 1): 2,
        })
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].shape, (3,))
        self.assertEqual(features[0][0], 1)
        self.assertEqual(features[0][1], 0)
        self.assertEqual(features[0][2], 0)

    def test_feature_extraction_multiple_frequencies(self):
        features = extract_features('This is a test', frequencies={
            ('This', 1): 1,
            ('is', 1): 0,
            ('a', 1): 1,
            ('test', 1): 2,
        })
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].shape, (3,))
        self.assertEqual(features[0][0], 1)
        self.assertEqual(features[0][1], 2)
        self.assertEqual(features[0][2], 0)

    def test_feature_extraction_no_stop_words(self):
        features = extract_features('Stocks fell 100 points on saturday', frequencies={
            ('stock', 1): 1,
            ('fell', 0): 1,
            ('point', 1): 2,
            ('saturday', 1): 0,
        })
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].shape, (3,))
        self.assertEqual(features[0][0], 1)
        self.assertEqual(features[0][1], 3)
        self.assertEqual(features[0][2], 1)