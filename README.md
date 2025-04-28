# Tweet/Post Moderation Classifier

A minimal end-to-end binary text classification pipeline that flags tweets as malicious/negative or benign/positive using TF-IDF + Logistic Regression.

---

## Files

- `sentiment140.csv`  
  Raw Sentiment140 dataset (cols: `target,id,date,flag,user,text`)
- `tweet_moderation.py`  
  Script with the full pipeline
- `requirements.txt`  
  Pinned Python dependencies
- `README.md`  
  This file

---

## Data Source

Download the Sentiment140 dataset from Kaggle and save it as `sentiment140.csv` in your project directory:  
https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download

---

## Setup

1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   python3 -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
