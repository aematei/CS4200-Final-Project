from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from tqdm import tqdm

def train_in_batches(X, y, model, batch_size=1000):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    for start_idx in tqdm(range(0, len(X), batch_size), desc="Training batches"):
        batch_indices = indices[start_idx:start_idx + batch_size]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        model.partial_fit(X_batch, y_batch, classes=np.unique(y))
    
    return model

def perform_cross_validation(model, X, y, n_folds=5):
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores

def tune_hyperparameters(model, param_grid, X, y):
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_