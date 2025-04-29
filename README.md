# Tweet/Post Moderation Classifier

An enhanced binary text classification pipeline that flags tweets as malicious/negative or benign/positive using TF-IDF vectorization and Logistic Regression with advanced features.

---

## Features

- Text preprocessing and normalization
- TF-IDF vectorization with n-gram support
- Hyperparameter tuning using GridSearchCV
- Cross-validation for model validation
- Batch processing support
- Comprehensive visualizations:
  - Learning curves
  - Confusion matrix
  - ROC curve
- Model persistence (save/load capability)

## Files

- `sentiment140.csv`  
  Raw Sentiment140 dataset (cols: `target,id,date,flag,user,text`)
- `tweet_moderation.py`  
  Main script with the classification pipeline
- `visualizations.py`  
  Visualization utilities for model evaluation
- `model_utils.py`  
  Model training and evaluation utilities
- `requirements.txt`  
  Python dependencies
- `README.md`  
  Project documentation

---

## Data Source

Download the Sentiment140 dataset from Kaggle and save it as `sentiment140.csv` in your project directory:  
https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download

---

## Setup

1. Create and activate a virtual environment (optional but recommended):
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Run the classifier:
   ```bash
   python tweet_moderation.py
   ```

## Output

The script will:
1. Load and preprocess the dataset
2. Split data into training (80%) and testing (20%) sets
3. Perform TF-IDF vectorization
4. Train model with optimized hyperparameters
5. Evaluate using cross-validation
6. Generate performance visualizations
7. Save the trained model for future use

## Model Performance

The classifier provides multiple evaluation metrics:
- Accuracy score
- Classification report (precision, recall, F1-score)
- Cross-validation scores
- Visual performance analysis through plots
