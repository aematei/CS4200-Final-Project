import unittest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tweet_moderation import preprocess_text, predict_sentiment

class TestTweetModeration(unittest.TestCase):
    
    def setUp(self):
        # Create a simple vectorizer and model for testing
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.model = LogisticRegression()
        
        # Create some training data
        texts = [
            "I love this product, it's amazing",
            "Great service and friendly staff",
            "This is terrible, I hate it",
            "Worst experience ever, avoid at all costs"
        ]
        labels = [1, 1, 0, 0]  # 1 = positive, 0 = negative
        
        # Fit the vectorizer and model
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
    
    def test_preprocess_text(self):
        """Test that text preprocessing works as expected"""
        self.assertEqual(preprocess_text("HELLO World!"), "hello world!")
        self.assertEqual(preprocess_text("  spaces  "), "spaces")
        self.assertEqual(preprocess_text(123), "123")  # Should handle non-string input
    
    def test_predict_sentiment_positive(self):
        """Test prediction for positive sentiment"""
        result = predict_sentiment("I really love this", self.vectorizer, self.model)
        self.assertEqual(result['prediction'], 1)
        self.assertEqual(result['label'], 'Positive/Benign')
        self.assertGreater(result['probability'], 0.5)
    
    def test_predict_sentiment_negative(self):
        """Test prediction for negative sentiment"""
        result = predict_sentiment("I really hate this", self.vectorizer, self.model)
        self.assertEqual(result['prediction'], 0)
        self.assertEqual(result['label'], 'Negative/Malicious')
        self.assertGreater(result['probability'], 0.5)

if __name__ == '__main__':
    unittest.main()