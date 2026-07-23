"""
NLP Preprocessing & Input Validation Module for TruthLens AI
Handles text cleaning, NLTK lemmatization, stopword removal,
and strict gibberish / non-news text detection.
"""

import re
import string
import nltk
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources quietly
for resource in ['stopwords', 'wordnet', 'punkt', 'words']:
    try:
        nltk.data.find(f'corpora/{resource}') if 'corpora' in resource else nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

lemmatizer = WordNetLemmatizer()

try:
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except Exception:
    ENGLISH_STOPWORDS = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "is", "was", "are", "were"}

try:
    ENGLISH_WORDS_SET = set(w.lower() for w in words.words())
except Exception:
    ENGLISH_WORDS_SET = set()

def clean_and_lemmatize(text):
    """
    Comprehensive NLP cleaning pipeline:
    1. Lowercase conversion
    2. Remove URLs & HTML tags
    3. Remove numbers & punctuation & special characters
    4. Remove extra spaces
    5. Remove NLTK stopwords
    6. Perform Lemmatization using NLTK WordNetLemmatizer
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Lowercase conversion
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove numbers & digits
    text = re.sub(r'\d+', '', text)
    
    # 5. Remove punctuation & special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'_', ' ', text)
    
    # 6. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Tokenize, remove stopwords, and lemmatize
    tokens = text.split()
    cleaned_tokens = []
    
    for token in tokens:
        if token not in ENGLISH_STOPWORDS and len(token) > 2:
            lemmatized = lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemmatized)
            
    return " ".join(cleaned_tokens)

def validate_news_input(text):
    """
    Strictly validates user text input to reject gibberish, keyboard smashing (abc, asdfgh, 123456, kfnkla),
    numbers only, special characters only, repeated characters, extremely short text, or empty input.
    Returns: (is_valid: bool, message: str)
    """
    if not text or not isinstance(text, str):
        return False, "Please enter a meaningful news article."
        
    trimmed = text.strip()
    words_list = trimmed.split()
    
    # Rule 1: Empty or extremely short input (< 10 words)
    if len(words_list) < 10:
        return False, "Please enter a meaningful news article."
        
    # Rule 2: Numbers only or special characters only
    letters_only = re.sub(r'[^a-zA-Z]', '', trimmed)
    if not letters_only or len(letters_only) < 25:
        return False, "Please enter a meaningful news article."
        
    # Rule 3: Repeated character sequences (e.g. 'aaaaaaa', 'asdfasdfasdf')
    if re.search(r'(.)\1{4,}', letters_only):
        return False, "Please enter a meaningful news article."
        
    # Rule 4: Keyboard smashing & unnatural word length
    avg_word_len = len(letters_only) / len(words_list)
    if avg_word_len > 13.5 or avg_word_len < 2.8:
        return False, "Please enter a meaningful news article."
        
    # Rule 5: Vowel ratio check (Natural English text has 30% - 50% vowels)
    vowels = set('aeiouAEIOU')
    vowel_count = sum(1 for c in letters_only if c in vowels)
    vowel_ratio = vowel_count / len(letters_only)
    
    if vowel_ratio < 0.18 or vowel_ratio > 0.68:
        return False, "Please enter a meaningful news article."
        
    # Rule 6: English dictionary lookup on first 12 words
    if ENGLISH_WORDS_SET:
        sample_tokens = [re.sub(r'[^a-zA-Z]', '', w).lower() for w in words_list[:12] if len(w) > 2]
        if sample_tokens:
            valid_count = sum(1 for w in sample_tokens if w in ENGLISH_WORDS_SET)
            valid_ratio = valid_count / len(sample_tokens)
            if valid_ratio < 0.35:
                return False, "Please enter a meaningful news article."
                
    return True, "Valid"

def calculate_reading_metadata(raw_text):
    """
    Calculate Word Count, Character Count, and Estimated Reading Time.
    """
    if not raw_text:
        return {'word_count': 0, 'char_count': 0, 'reading_time_mins': 0}
        
    char_count = len(raw_text)
    words_list = raw_text.strip().split()
    word_count = len(words_list)
    reading_time_mins = max(1, round(word_count / 200)) if word_count > 0 else 0
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'reading_time_mins': reading_time_mins
    }
