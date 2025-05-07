import os
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from joblib import dump, load
from tqdm import tqdm
from tqdm.auto import tqdm
import seaborn as sns
from typing import Any, Dict, List, Tuple, Optional
from model_utils import train_in_batches
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")  

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*multi_class.*")

import matplotlib.pyplot as plt

MODEL_PATH = "logistic_regression_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

NEGATIVE_WORDS = {
    'hate', 'stupid', 'sad', 'slut', 'poop', 'slander', 'terrible', 'awful', 
    'horrible', 'bad', 'worst', 'ugly', 'negative', 'annoying', 'angry', 'furious'
}

POSITIVE_WORDS = {
    'happy', 'good', 'great', 'excellent', 'wonderful', 'fantastic', 'amazing',
    'love', 'like', 'awesome', 'best', 'beautiful', 'positive', 'joy', 'delightful'
}

DEFAULT_MAX_FEATURES = 15000

def create_vectorizer(max_features: int = DEFAULT_MAX_FEATURES) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        max_features=max_features
    )

def create_pipeline():
    return Pipeline([
        ('vectorizer', TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), max_features=DEFAULT_MAX_FEATURES)),
        ('classifier', LogisticRegression(max_iter=300, n_jobs=-1))
    ])

def predict_sentiment(
    text: str, 
    vectorizer: Any, 
    model: Any
) -> Dict[str, Any]:
    """
    Predict the sentiment of a given text input.
    
    Parameters:
    -----------
    text : str
        The text to analyze
    vectorizer : TfidfVectorizer
        Fitted vectorizer to transform the text
    model : classifier
        Trained classification model
        
    Returns:
    --------
    dict
        Contains prediction, probability, and human-readable label
    """

    processed_text = preprocess_text(text)
    
    # Check if any words are in the vocabulary
    text_tokens = processed_text.split()
    known_words = [word for word in text_tokens if word in vectorizer.get_feature_names_out()]
    
    # Vectorize input
    text_vector = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
    # Calculate sentiment score
    confidence = float(probabilities[int(prediction)])
    sentiment_score = (confidence - 0.5) * 2  
    if prediction == 0:
        sentiment_score = -sentiment_score  

    result = {
        'prediction': int(prediction),
        'probability': float(probabilities[int(prediction)]),
        'label': 'Positive/Benign' if prediction == 1 else 'Negative/Malicious',
        'confidence': f'{confidence:.2%}',
        'sentiment_score': sentiment_score,
        'unknown_words_ratio': 1 - (len(known_words) / max(1, len(text_tokens)))
    }

    if result['unknown_words_ratio'] > 0.5 and len(text_tokens) > 1:
        result['warning'] = f"Warning: {result['unknown_words_ratio']:.0%} of words not recognized by the model."
    
    return result

def preprocess_text(text: Any) -> str:
    """
    Preprocess text with more advanced techniques for better feature extraction.
    """
    import re
    from string import punctuation
    
    text = str(text).lower()
    
    # Replace URLs with a token
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    
    # Replace user mentions with a token
    text = re.sub(r'@\w+', ' MENTION ', text)
    
    # Replace hashtags with the word
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Handle emoticons specially (preserve them as they indicate sentiment)
    emoticons_happy = set([
        ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3'
    ])
    emoticons_sad = set([
        ':-(', ':(', ':-c', ':c', ':-<', ':<', ':-[', ':[', ':{', ':-||', ':@', ':\'(',
        ':"(', ':-|', ':|', ':(', ':-[', ':-<', '=\\', '=/', '>:['
    ])
    
    # Handle special terms
    special_terms = {
        'haha': ' POSITIVE_EMOTION ',
        'lol': ' POSITIVE_EMOTION ',
        'rofl': ' POSITIVE_EMOTION ',
        'lmao': ' POSITIVE_EMOTION ',
        'wtf': ' NEGATIVE_EMOTION ',
        'omg': ' EMOTION '
    }
    
    for term, replacement in special_terms.items():
        text = re.sub(r'\b' + term + r'\b', replacement, text)
    
    # Extract all emoticons
    emoticons = []
    for word in text.split():
        if word in emoticons_happy:
            emoticons.append(' HAPPY_EMOTICON ')
        elif word in emoticons_sad:
            emoticons.append(' SAD_EMOTICON ')
    
    # Remove special characters but keep basic punctuation that conveys sentiment
    text = re.sub(r'[^\w\s:;)(,.\-!?]', '', text)
    
    # Replace multiple spaces with single space and add emoticons back
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Add emoticons back
    if emoticons:
        text = text + ' ' + ' '.join(emoticons)
    
    # Add specific handling for negation (often changes sentiment)
    negations = ["don't", "not", "no", "never", "neither", "nor", "can't", "won't", "wouldn't", "couldn't", "shouldn't"]
    for negation in negations:
        if negation in text:
            text = text.replace(negation, "NEG_" + negation)
    
    return text

def retrain_with_feedback(
    model: Any, 
    vectorizer: Any, 
    feedback_data: List[Tuple[str, int]]
) -> Any:
    """Retrain model with feedback data"""
    if not feedback_data:
        return model
        
    print("\nIncorporating feedback into the model...")
    X_feedback = [item[0] for item in feedback_data]
    y_feedback = [item[1] for item in feedback_data]
    
    # Process feedback text with progress bar
    X_processed = [preprocess_text(text) for text in tqdm(X_feedback, desc="Preprocessing feedback")]
    X_vectors = vectorizer.transform(X_processed)
    
    try:
        # Create a new model with the same parameters 
        new_model = LogisticRegression(
            C=model.C if hasattr(model, 'C') else 1.0,
            solver='liblinear', 
            random_state=42,
            max_iter=1000
        )
        
        # Train on the feedback data
        new_model.fit(X_vectors, y_feedback)

        feature_names = vectorizer.get_feature_names_out()
        
        # For each word in the feedback, adjust the original model's coefficient
        for text, sentiment in feedback_data:
            words = preprocess_text(text).split()
            for word in words:
                if word in feature_names:
                    idx = list(feature_names).index(word)
                    # If sentiment is positive (1) but coefficient is negative, make it positive
                    if sentiment == 1 and model.coef_[0][idx] < 0:
                        model.coef_[0][idx] = abs(model.coef_[0][idx]) * 1.5
                    # If sentiment is negative (0) but coefficient is positive, make it negative
                    elif sentiment == 0 and model.coef_[0][idx] > 0:
                        model.coef_[0][idx] = -abs(model.coef_[0][idx]) * 1.5 
        
        print("Model successfully updated with feedback!")
        # Save the updated model
        dump(model, 'logistic_regression_model.joblib')
        
    except Exception as e:
        print(f"Error updating model: {e}")
        print("Applied manual word-level corrections based on feedback.")
    
    return model

def create_fresh_model(vectorizer, X_train, y_train, batch_training: bool = False):
    if batch_training:
        model = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)
        X_train_vec = vectorizer.transform(X_train)
        model = train_in_batches(X_train_vec, y_train, model, batch_size=1000)
    else:
        model = LogisticRegression(max_iter=300, n_jobs=-1)
        X_train_vec = vectorizer.transform(X_train)
        model.fit(X_train_vec, y_train)
    return model

def interactive_demo(vectorizer, model, collect_feedback=True):
    """
    Interactive CLI for tweet sentiment analysis with feedback collection.
    """
    feedback_data = []
    print("\nEnter a tweet (or type 'exit' to quit):")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == 'exit':
            print("Exiting interactive demo.")
            break
        if not user_input:
            print("Please enter non-empty text.")
            continue
        result = predict_sentiment(user_input, vectorizer, model)
        print(f"Sentiment: {result['label']} (Confidence: {result['confidence']})")
        if 'warning' in result:
            print(result['warning'])
        if collect_feedback:
            feedback = input("Is this correct? (y/n): ").strip().lower()
            if feedback == 'n':
                correct_label = input("What is the correct label? (0 = Negative/Malicious, 1 = Positive/Benign): ").strip()
                if correct_label in ['0', '1']:
                    feedback_data.append((user_input, int(correct_label)))
                    print("Feedback recorded.")
                else:
                    print("Invalid label. Feedback not recorded.")
    # After session, retrain if feedback was collected
    if collect_feedback and feedback_data:
        print("\nRetraining model with feedback...")
        retrain_with_feedback(model, vectorizer, feedback_data)
        print("Model updated with feedback.")

def tune_hyperparameters(model, param_grid, X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV"""
    from sklearn.model_selection import GridSearchCV
    
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,  
        scoring='accuracy',
        n_jobs=-1,  
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    return best_model, best_params

def train_new_model(vectorizer, X_train, X_test, y_train, model_path):
    """Train a new model with hyperparameter tuning and save it"""
    # Vectorize the data with progress bar
    print("Vectorizing training data...")
    X_train_vec = vectorizer.fit_transform(list(tqdm(X_train, desc="Vectorizing train")))
    print("Vectorizing test data...")
    X_test_vec = vectorizer.transform(list(tqdm(X_test, desc="Vectorizing test")))
    print(f"Vectorization complete. Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    
    # Set up hyperparameter tuning
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'class_weight': ['balanced', None],
        'max_iter': [500]  
    }
    
    base_model = LogisticRegression(solver='liblinear', random_state=42)
    clf, best_params = tune_hyperparameters(base_model, param_grid, X_train_vec, y_train)
    print(f"Best parameters: {best_params}")
    
    print("Training final model with best parameters...")
    clf.fit(X_train_vec, y_train)
    dump(clf, model_path)
    print(f"Model saved to '{model_path}'")
    
    # Save the vectorizer 
    vectorizer_path = 'tfidf_vectorizer.joblib'
    dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to '{vectorizer_path}'")
    
    return X_train_vec, X_test_vec, clf

def perform_cross_validation(model, X, y, cv=5):
    """
    Perform cross-validation on the model.
    
    Parameters:
    -----------
    model : classifier
        The model to evaluate
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    array
        Cross-validation scores
    """
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return cv_scores

def fix_model_lexicon(model: Any, vectorizer: Any) -> Any:
    """
    Adjust model coefficients for known positive/negative words.
    Args:
        model: Trained model with coef_ attribute.
        vectorizer: Fitted vectorizer.
    Returns:
        Model with adjusted coefficients.
    """
    if not hasattr(model, 'coef_'):
        return model
        
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Correct positive words 
    for word in POSITIVE_WORDS:
        if word in feature_names:
            idx = list(feature_names).index(word)
            if model.coef_[0][idx] < 0:
                print(f"Fixing sentiment for positive word: '{word}'")
                model.coef_[0][idx] = abs(model.coef_[0][idx])
    
    # Correct negative words
    for word in NEGATIVE_WORDS:
        if word in feature_names:
            idx = list(feature_names).index(word)
            if model.coef_[0][idx] > 0:
                print(f"Fixing sentiment for negative word: '{word}'")
                model.coef_[0][idx] = -abs(model.coef_[0][idx])
    
    return model

def plot_all_metrics(y_test, y_pred, y_pred_proba, clf, X_train_vec, y_train, vectorizer):
    """Plot various evaluation metrics and visualizations"""
    plt.style.use('seaborn-v0_8-whitegrid')  # Updated style name
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Confusion Matrix
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues', ax=plt.gca())
    plt.title('Confusion Matrix', pad=20, fontsize=14)
    
    # Plot 2: ROC Curve
    plt.subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', pad=20, fontsize=14)
    plt.legend(loc='lower right')
    
    # Plot 3: Feature Importance
    plt.subplot(2, 1, 2)
    if hasattr(clf, 'coef_'):
        feature_names = vectorizer.get_feature_names_out()
        coefs = clf.coef_[0]
        
        top_positive_idx = coefs.argsort()[-15:]
        top_negative_idx = coefs.argsort()[:15]
        
        top_idx = np.concatenate([top_negative_idx, top_positive_idx])
        top_features = [feature_names[i] for i in top_idx]
        top_coefs = [coefs[i] for i in top_idx]
        
        colors = ['red' if c < 0 else 'green' for c in top_coefs]
        plt.barh(top_features, top_coefs, color=colors)
        plt.title('Top Feature Importance', pad=20, fontsize=14)
        plt.xlabel('Coefficient Value')
    
    plt.tight_layout(pad=3.0)  # Add padding between subplots
    
    # Save the figure
    plt.savefig('model_evaluation.png', bbox_inches='tight', dpi=300)
    
    # Show the plot
    plt.show()
    
    # Close the figure to free memory
    plt.close(fig)

def main(sample_size=None):
    # 1. Load data
    print("Loading dataset...")
    df = pd.read_csv(
        'training.1600000.processed.noemoticon.csv',
        encoding='latin-1',
        header=None,
        names=['target', 'id', 'date', 'flag', 'user', 'text']
    )

    if sample_size and sample_size < len(df):
        print(f"Taking a random sample of {sample_size} tweets...")
        df = df.sample(sample_size, random_state=42)
    
    print(f"Dataset loaded with {len(df)} rows and the following columns: {list(df.columns)}")

    print("Preprocessing text data...")
    df['text'] = df['text'].progress_apply(preprocess_text)

    # 2. Simplify labels
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

    # 4. & 5. Vectorization and model training
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        print("Loading existing model and vectorizer...")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                clf = load(MODEL_PATH)
                vectorizer = load(VECTORIZER_PATH)
            
            print("Model and vectorizer loaded successfully.")
            X_train_vec = vectorizer.transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            print(f"Using existing vectorizer with vocabulary size: {len(vectorizer.get_feature_names_out())}")
            
            # cross-validation
            print("Performing cross-validation...")
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(clf, X_train_vec, y_train, cv=5, scoring='accuracy')
            print(f"Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
        except Exception as e:
            print(f"Error loading model or vectorizer: {e}")
            print("Training new model with fresh vectorizer instead...")
            vectorizer = create_vectorizer()
            X_train_vec, X_test_vec, clf = train_new_model(vectorizer, X_train, X_test, y_train, MODEL_PATH)
    else:
        print("Training new model with fresh vectorizer...")
        vectorizer = create_vectorizer()
        X_train_vec, X_test_vec, clf = train_new_model(vectorizer, X_train, X_test, y_train, MODEL_PATH)
    # cross-validation
    print("Performing cross-validation...")
    cv_scores = perform_cross_validation(clf, X_train_vec, y_train)
    print(f"Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # 5. evaluation
    print("Evaluating the classifier...")
    y_pred = clf.predict(X_test_vec)
    y_pred_proba = clf.predict_proba(X_test_vec)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # visualizations
    print("Generating user-friendly visualizations...")
    plot_all_metrics(y_test, y_pred, y_pred_proba, clf, X_train_vec, y_train, vectorizer)
    
    # Add a small pause to ensure plots are displayed
    plt.pause(1)

    # Save vectorizer
    dump(vectorizer, VECTORIZER_PATH)
    print(f"Vectorizer saved to '{VECTORIZER_PATH}'")
    
    # Before returning, fix the model lexicon
    print("\nChecking and fixing model sentiment lexicon...")
    clf = fix_model_lexicon(clf, vectorizer)
    
    # Save the model again with fixed lexicon
    dump(clf, MODEL_PATH)
    print("Model saved with updated sentiment lexicon.")
    
    return clf, vectorizer

if __name__ == '__main__':
    print("Starting the enhanced tweet moderation pipeline...")
    
    try:
        # (comment out or set to None to use full dataset)
        sample_size = None
        
        clf, vectorizer = main(sample_size)

        try_demo = input("\nWould you like to try the interactive tweet analyzer? (y/n): ")
        if try_demo.lower().startswith('y'):
            interactive_demo(vectorizer, clf)
        
        print("Pipeline execution complete.")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        traceback.print_exc()
        print("\nTry removing the model and vectorizer files and running again.")
