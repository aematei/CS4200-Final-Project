import pandas as pd
from tweet_moderation import main, predict_sentiment, interactive_demo
from joblib import load
import os

def run_demo():
    print("="*70)
    print("             TWEET SENTIMENT ANALYSIS DEMO")
    print("="*70)

    # Step 1: Train/load the model
    print("\n1. Initializing the model...")
    if os.path.exists('logistic_regression_model.joblib') and os.path.exists('vectorizer.joblib'):
        print("Loading existing model and vectorizer...")
        model = load('logistic_regression_model.joblib')
        vectorizer = load('vectorizer.joblib')
    else:
        print("Training new model...")
        model, vectorizer = main(sample_size=10000)  # Use smaller sample for demo

    # Step 2: Example predictions
    print("\n2. Testing with example tweets...")
    example_tweets = [
        "I absolutely love this product! Best purchase ever!",
        "This is terrible, worst experience of my life.",
        "The weather is nice today.",
        "I'm really disappointed with the service.",
        "Just got a promotion at work! So excited!"
    ]

    for tweet in example_tweets:
        result = predict_sentiment(tweet, vectorizer, model)
        print("\nTweet:", tweet)
        print("Sentiment:", result['label'])
        # Fixed the confidence formatting
        print("Confidence:", result['confidence'])
        print("-" * 50)

    # Step 3: Interactive mode
    print("\n3. Starting interactive mode...")
    print("You can now enter your own tweets to analyze!")
    interactive_demo(vectorizer, model)

if __name__ == "__main__":
    run_demo()