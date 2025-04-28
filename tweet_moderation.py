#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load

def main():
    # 1. Load data
    print("Loading dataset...")
    df = pd.read_csv(
        'training.1600000.processed.noemoticon.csv',
        encoding='latin-1',
        header=None,
        names=['target', 'id', 'date', 'flag', 'user', 'text']
    )
    print(f"Dataset loaded with {len(df)} rows and the following columns: {list(df.columns)}")

    # 2. Simplify labels → 0 (negative/malicious) or 1 (positive/benign)
    print("Simplifying labels...")
    df['label'] = df['target'].apply(lambda x: 0 if x == 0 else 1)
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # 3. Train/test split
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

    # 4. Vectorize tweets
    print("Vectorizing text data using TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vectorization complete. Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    # 5. Train or load classifier
    model_path = 'logistic_regression_model.joblib'

    if os.path.exists(model_path):
        print("Loading existing model...")
        clf = load(model_path)
    else:
        print("Training new model...")
        clf = LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            max_iter=200
        )
        clf.fit(X_train_vec, y_train)
        dump(clf, model_path)
        print(f"Model saved to '{model_path}'")

    # 6. Evaluate
    print("Evaluating the classifier...")
    y_pred = clf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == '__main__':
    print("Starting the tweet moderation pipeline...")
    main()
    print("Pipeline execution complete.")
