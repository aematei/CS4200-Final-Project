# Tweet/Post Moderation Classifier

A minimal end-to-end binary text classification pipeline that flags tweets as malicious/negative or benign/positive using TF-IDF + Logistic Regression.

Dataset taken from https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download

---

## Files

- `sentiment140.csv`  
  Raw Sentiment140 dataset (cols: `target,id,date,flag,user,text`)
- `tweet_moderation.ipynb`  
  Single-cell Jupyter notebook with the full pipeline
- `tweet_moderation.py` _(optional)_  
  Script version of the notebook
- `README.md`  
  This file

---

## Requirements

- **Python** ≥ 3.8  
- **pip** (for package installation)

Install dependencies:
```bash
pip install pandas scikit-learn jupyter
