import os
import sys
import json
import pickle
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Import preprocessing module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.preprocessing import clean_and_lemmatize

def load_dataset(base_dir):
    """
    Load Real and Fake news datasets.
    """
    true_path = os.path.join(base_dir, 'dataset', 'True.csv')
    fake_path = os.path.join(base_dir, 'dataset', 'Fake.csv')
    fallback_path = os.path.join(base_dir, 'dataset', 'fake_or_real_news.csv')
    
    if os.path.exists(true_path) and os.path.exists(fake_path):
        print(f"[*] Loading True news dataset from: {true_path}")
        df_true = pd.read_csv(true_path)
        df_true['label'] = 'REAL'
        
        print(f"[*] Loading Fake news dataset from: {fake_path}")
        df_fake = pd.read_csv(fake_path)
        df_fake['label'] = 'FAKE'
        
        df = pd.concat([df_true, df_fake], ignore_index=True)
    elif os.path.exists(fallback_path):
        print(f"[*] Loading backup news dataset from: {fallback_path}")
        df = pd.read_csv(fallback_path)
        df['label'] = df['label'].astype(str).str.upper()
    else:
        raise FileNotFoundError("No dataset files found in dataset/ directory.")
        
    df['combined_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
    return df

def train_and_compare_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    print("\n========================================================")
    print(" [+] TRUTHLENS AI - MULTI-MODEL ML TRAINING PIPELINE")
    print("========================================================\n")
    
    # 1. Load Dataset
    df = load_dataset(base_dir)
    print(f"[+] Total Dataset Articles: {len(df)}")
    print(f"[+] Class Distribution:\n{df['label'].value_counts()}\n")
    
    # 2. NLP Preprocessing & Lemmatization
    print("[*] Applying NLTK cleaning, stopword removal, and lemmatization...")
    df['clean_content'] = df['combined_text'].apply(clean_and_lemmatize)
    df = df[df['clean_content'].str.strip() != ''].reset_index(drop=True)
    
    X = df['clean_content']
    y = df['label']
    
    # 3. Split Dataset (80% Train, 20% Test, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Split: {len(X_train)} Training samples, {len(X_test)} Testing samples")
    
    # 4. TF-IDF Vectorizer Setup
    print("[*] Configuring TF-IDF Vectorizer (max_features=10000, ngram_range=(1,2), min_df=2, max_df=0.9)...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 5. Define Candidate Models
    candidate_models = {
        'Logistic Regression': LogisticRegression(C=2.0, max_iter=1000, random_state=42),
        'Linear SVM': CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42)),
        'Passive Aggressive Classifier': CalibratedClassifierCV(PassiveAggressiveClassifier(max_iter=1000, random_state=42)),
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.1)
    }
    
    best_model_name = None
    best_model_obj = None
    best_f1 = -1.0
    best_metrics = {}
    
    print("\n--------------------------------------------------------")
    print(" EVALUATING CANDIDATE ALGORITHMS")
    print("--------------------------------------------------------")
    
    for name, clf in candidate_models.items():
        print(f"\n[*] Training {name}...")
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, pos_label='FAKE'))
        rec = float(recall_score(y_test, y_pred, pos_label='FAKE'))
        f1 = float(f1_score(y_test, y_pred, pos_label='FAKE'))
        cm = confusion_matrix(y_test, y_pred, labels=['REAL', 'FAKE']).tolist()
        cr = classification_report(y_test, y_pred)
        
        print(f"    - Accuracy:  {acc * 100:.2f}%")
        print(f"    - Precision: {prec * 100:.2f}%")
        print(f"    - Recall:    {rec * 100:.2f}%")
        print(f"    - F1 Score:  {f1 * 100:.2f}%")
        print(f"    - Confusion Matrix:\n      {cm}")
        print(f"    - Classification Report:\n{cr}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = clf
            best_metrics = {
                'model_name': name,
                'accuracy': round(acc * 100, 2),
                'precision': round(prec * 100, 2),
                'recall': round(rec * 100, 2),
                'f1_score': round(f1 * 100, 2),
                'confusion_matrix': cm,
                'classification_report': cr
            }
            
    print("\n--------------------------------------------------------")
    print(f"[+] HIGHEST PERFORMING MODEL: {best_model_name} (F1 Score: {best_f1 * 100:.2f}%)")
    print("--------------------------------------------------------\n")
    
    # 6. Save Model & Vectorizer Artifacts
    pkl_model_file = os.path.join(model_dir, 'model.pkl')
    pkl_vectorizer_file = os.path.join(model_dir, 'vectorizer.pkl')
    
    joblib_model_file = os.path.join(model_dir, 'news_classifier.joblib')
    joblib_vectorizer_file = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
    metrics_file = os.path.join(model_dir, 'metrics.json')
    
    # Save as .pkl
    with open(pkl_model_file, 'wb') as f:
        pickle.dump(best_model_obj, f)
    with open(pkl_vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    # Save as .joblib for compatibility
    joblib.dump(best_model_obj, joblib_model_file)
    joblib.dump(vectorizer, joblib_vectorizer_file)
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(best_metrics, f, indent=4)
        
    print(f"[OK] Saved best model to: {pkl_model_file}")
    print(f"[OK] Saved vectorizer to: {pkl_vectorizer_file}")
    print(f"[OK] Saved metrics to: {metrics_file}\n")
    
    return best_metrics

if __name__ == '__main__':
    train_and_compare_models()
