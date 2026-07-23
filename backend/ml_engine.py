"""
Machine Learning Engine Module for TruthLens AI
Contains utility functions for preprocessing text, feature extraction, and model loading.
"""

import re
import os
import joblib

def clean_text(text):
    """
    Sanitize text input by removing HTML tags, URLs, numbers, and special characters.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def load_saved_model(model_path, vectorizer_path):
    """
    Load joblib artifacts for ML classifier and TFIDF vectorizer.
    """
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    return None, None
