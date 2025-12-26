import unittest

from trading_sentiment_analysis.process.text import process_text_to_words


class ProcessTestCase(unittest.TestCase):
    def test_process_text_to_words(self):
        text = "This is a test! where #hashing is $ticker http://example.com"
        # Stemmer is aggressive and turns "this" to "thi"
        expected_output = ['test', 'hashing', 'ticker']
        output = process_text_to_words(text)
        self.assertEqual(output, expected_output)

    def test_process_text_to_words_empty(self):
        text = ""
        expected_output = []
        output = process_text_to_words(text)
        self.assertEqual(output, expected_output)

    def test_process_text_to_words_no_stopwords(self):
        text = "Hello World! This is a test."
        expected_output = ['hello', 'world', 'test']
        output = process_text_to_words(text)
        self.assertEqual(output, expected_output)

    def test_process_text_to_words_no_numbers(self):
        text = "Parke Bancorp, Inc. Q4 EPS $0.63 Up From $0.36"
        expected_output = ['parke', 'bancorp', 'inc', 'q4', 'eps', 'up', 'from']
        output = process_text_to_words(text)
        self.assertEqual(output, expected_output)

    def test_process_text_to_words_no_acronyms(self):
        text = "Parke Bancorp, Inc. Q4 EPS $0.63 Up From $0.36 🚀 #finance https://link.com"
        expected_output = ['parke', 'bancorp', 'inc', 'q4', 'eps', 'up', 'from', '🚀', 'finance']
        output = process_text_to_words(text)
        self.assertEqual(output, expected_output)