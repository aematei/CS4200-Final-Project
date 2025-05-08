from typing import Any, Tuple, Dict
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from tqdm import tqdm
from src.config import DATA_PATH, MODEL_PATH, VECTORIZER_PATH

def train_in_batches(
    X: Any, 
    y: Any, 
    model: Any, 
    batch_size: int = 1000
) -> Any:
    """
    Train a model in batches using partial_fit.
    Args:
        X: Feature matrix.
        y: Labels.
        model: Model supporting partial_fit.
        batch_size: Size of each batch.
    Returns:
        Trained model.
    """
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    for start_idx in tqdm(range(0, len(X), batch_size), desc="Training batches"):
        batch_indices = indices[start_idx:start_idx + batch_size]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        model.partial_fit(X_batch, y_batch, classes=np.unique(y))
    
    return model

def perform_cross_validation(
    model: Any, 
    X: Any, 
    y: Any, 
    n_folds: int = 5
) -> np.ndarray:
    """
    Perform cross-validation and return accuracy scores.
    Args:
        model: Model to evaluate.
        X: Feature matrix.
        y: Labels.
        n_folds: Number of folds.
    Returns:
        Array of cross-validation scores.
    """
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores

def tune_hyperparameters(
    model: Any, 
    param_grid: Dict, 
    X: Any, 
    y: Any
) -> Tuple[Any, Dict]:
    """
    Tune hyperparameters using GridSearchCV.
    Args:
        model: Model to tune.
        param_grid: Grid of parameters.
        X: Feature matrix.
        y: Labels.
    Returns:
        Tuple of (best_estimator, best_params).
    """
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_