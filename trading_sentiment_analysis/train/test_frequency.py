import unittest

from trading_sentiment_analysis.train.frequency import build_freqs, build_freqs_docs

def process_text_to_words(text):
    """A mock function to simulate the text processing."""
    # This is a placeholder for the actual text processing logic
    # For simplicity, we will just return the words in lowercase
    return [word.lower() for word in text.split() if word.isalpha()]

class BuildFreqsTestCase(unittest.TestCase):
    def test_build_freqs(self):
        # Test with a simple list of words
        texts = [
            "this is a test",
            "this is another test",
            "this is yet another test",
        ]
        expected_output = {
            ('this', 1): 3,
            ('is', 1): 3,
            ('a', 1): 1,
            ('test', 1): 3,
            ('another', 1): 2,
            ('yet', 1): 1,
        }
        output = build_freqs(texts, [1, 1, 1], word_processor=process_text_to_words)
        self.assertEqual(output, expected_output)

    def test_build_freqs_empty(self):
        # Test with an empty list
        words = []
        expected_output = {}
        output = build_freqs(words, [], word_processor=process_text_to_words)
        self.assertEqual(output, expected_output)

    def test_build_freqs_other_labels(self):
        # Test with a list of words and different labels
        texts = [
            "this is a test",
            "this is another test",
            "this is yet another test",
        ]
        expected_output = {
            ('this', 0): 1,
            ('this', 1): 2,
            ('is', 0): 1,
            ('is', 1): 2,
            ('a', 1): 1,
            ('test', 0): 1,
            ('test', 1): 2,
            ('another', 0): 1,
            ('another', 1): 1,
            ('yet', 1): 1,
        }
        output = build_freqs(texts, [1, 0, 1], word_processor=process_text_to_words)
        self.assertEqual(output, expected_output)

class BuildFreqsDocsTestCase(unittest.TestCase):
    def test_build_freqs_docs(self):
        # Test with a simple list of words
        texts = [
            "this is a test",
            "this is another test",
            "this is yet another test",
        ]
        labels = [1, 1, 1]
        expected_freqs = {
            ('this', 1): 3,
            ('is', 1): 3,
            ('a', 1): 1,
            ('test', 1): 3,
            ('another', 1): 2,
            ('yet', 1): 1,
        }
        expected_doc_freqs = {
            'this': 3,
            'is': 3,
            'a': 1,
            'test': 3,
            'another': 2,
            'yet': 1,
        }
        expected_N = 3

        freqs, doc_freqs, N = build_freqs_docs(texts, labels, word_processor=process_text_to_words)
        print(freqs)
        print(doc_freqs)
        print(N)
        self.assertEqual(freqs, expected_freqs)
        self.assertEqual(doc_freqs, expected_doc_freqs)
        self.assertEqual(N, expected_N)
        
    def test_build_freqs_docs_multiple_labels(self):
        # Test with a list of words and different labels
        texts = [
            "this is a test",
            "this is another test",
            "this is yet another test",
        ]
        labels = [1, 0, 1]
        expected_freqs = {
            ('this', 1): 2,
            ('this', 0): 1,
            ('is', 1): 2,
            ('is', 0): 1,
            ('a', 1): 1,
            ('test', 1): 2,
            ('test', 0): 1,
            ('another', 1): 1,
            ('another', 0): 1,
            ('yet', 1): 1,
        }
        expected_doc_freqs = {
            'this': 3,
            'is': 3,
            'a': 1,
            'test': 3,
            'another': 2,
            'yet': 1,
        }
        expected_N = 3

        freqs, doc_freqs, N = build_freqs_docs(texts, labels, word_processor=process_text_to_words)
        self.assertEqual(freqs, expected_freqs)
        self.assertEqual(doc_freqs, expected_doc_freqs)
        self.assertEqual(N, expected_N)