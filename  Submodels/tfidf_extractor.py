import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFExtractor:
    def __init__(self, texts):
        """
        texts: list of all texts used to fit the TF-IDF vectorizer
        """
        self.vectorizer = TfidfVectorizer(max_features=512)
        self.vectorizer.fit(texts)

    def extract(self, text):
        """
        text: single string or list of strings
        Returns numpy array of tf-idf features
        """
        if isinstance(text, list):
            return self.vectorizer.transform(text).toarray()
        else:
            return self.vectorizer.transform([text]).toarray().flatten()
