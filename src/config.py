import os

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(BASE_DIR, "data", "training.1600000.processed.noemoticon.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_regression_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.joblib")