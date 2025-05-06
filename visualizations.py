import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

def plot_learning_curve(estimator, X, y, title='Learning Curve'):
    """Generate a learning curve with improved clarity and explanations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy')
    
    plt.figure(figsize=(12, 10))
    
    # Calculate statistics
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot average scores with filled std
    plt.plot(train_sizes, train_mean, 'o-', color='#1f77b4', 
             label='Training score', linewidth=2.5, markersize=8)
    plt.plot(train_sizes, test_mean, 'o-', color='#ff7f0e', 
             label='Cross-validation score', linewidth=2.5, markersize=8)
    
    plt.fill_between(train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1, color='#1f77b4')
    plt.fill_between(train_sizes, test_mean - test_std, 
                     test_mean + test_std, alpha=0.1, color='#ff7f0e')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Number of Training Examples', fontsize=14)
    plt.ylabel('Accuracy Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower right', fontsize=12)

    gap = train_mean[-1] - test_mean[-1]
    gap_text = f"Gap: {gap:.2f}"
    
    if gap > 0.05:
        insight = "Model shows overfitting"
    elif test_mean[-1] < 0.6:
        insight = "Model may need more complexity"
    else:
        insight = "Model shows good generalization"

    plt.annotate(f"{insight}\n{gap_text}",
                xy=(0.7, 0.25), xycoords='figure fraction',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", alpha=0.1))

    explanation = (
        "Learning Curve Interpretation:\n"
        "• Converging lines = Model has enough data\n"
        "• Wide gap between lines = Overfitting\n"
        "• Low test score = Underfitting\n"
        "• Ideal: High test score with small gap to training score"
    )
    
    plt.figtext(0.5, 0.01, explanation, fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.5", alpha=0.1),
               ha='center')
    
    plt.grid(True)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])  
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=['Negative', 'Positive']):
    """Plot a confusion matrix to show model classification performance with clearer labels."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10)) 
    
    # Calculate metrics for annotation
    total = np.sum(cm)
    accuracy = np.trace(cm) / total
    misclassification = 1 - accuracy

    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix: Tweet Classification Results', fontsize=16, fontweight='bold')
    plt.ylabel('True Sentiment', fontsize=14)
    plt.xlabel('Predicted Sentiment', fontsize=14)

    plt.figtext(0.5, 0.01, f'Accuracy: {accuracy:.2%} | Misclassification: {misclassification:.2%}', 
               ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", alpha=0.1))

    cell_labels = [
        f"True Negatives\n({cm[0,0]})\nCorrectly identified\nnegative tweets",
        f"False Positives\n({cm[0,1]})\nNegative tweets\nmisclassified as positive",
        f"False Negatives\n({cm[1,0]})\nPositive tweets\nmisclassified as negative",
        f"True Positives\n({cm[1,1]})\nCorrectly identified\npositive tweets"
    ]

    plt.annotate(cell_labels[0], xy=(0.18, 0.82), xycoords='figure fraction',
                fontsize=11, ha='center', bbox=dict(boxstyle="round,pad=0.5", alpha=0.1))
    
    plt.annotate(cell_labels[1], xy=(0.82, 0.82), xycoords='figure fraction',
                fontsize=11, ha='center', bbox=dict(boxstyle="round,pad=0.5", alpha=0.1))
    
    plt.annotate(cell_labels[2], xy=(0.18, 0.18), xycoords='figure fraction',
                fontsize=11, ha='center', bbox=dict(boxstyle="round,pad=0.5", alpha=0.1))
    
    plt.annotate(cell_labels[3], xy=(0.82, 0.18), xycoords='figure fraction',
                fontsize=11, ha='center', bbox=dict(boxstyle="round,pad=0.5", alpha=0.1))
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.93])
    plt.show()

def plot_roc_curve(y_test, y_pred_proba):
    """Plot ROC curve with improved explanations and layout."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(12, 10)) 
    
    # Main ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier (AUC = 0.5)')

    threshold_points = [0.3, 0.5, 0.7]
    annotations = []
    
    for i, threshold in enumerate(threshold_points):
        idx = (np.abs(thresholds - threshold)).argmin()
        if idx < len(fpr):
            plt.plot(fpr[idx], tpr[idx], 'ro', markersize=8)

            xytext_offset = [(30, -20), (20, 30), (-30, -40)][i % 3]
            
            annot = plt.annotate(f'threshold={threshold:.1f}', 
                        (fpr[idx], tpr[idx]), 
                        xytext=xytext_offset, 
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                        bbox=dict(boxstyle="round,pad=0.3", alpha=0.1))
            annotations.append(annot)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)

    explanation = (
        "ROC Curve Interpretation:\n"
        "• Closer to top-left corner = better performance\n"
        "• AUC of 1.0 = perfect classifier\n"
        "• AUC of 0.5 = random guessing (diagonal line)\n"
        "• Higher threshold = fewer false positives but more false negatives"
    )

    plt.figtext(0.5, 0.01, explanation, fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.5", alpha=0.1),
               ha='center')
    
    plt.grid(True)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95]) 
    plt.show()

def plot_precision_recall_curve(y_test, y_pred_proba):
    """Plot precision-recall curve with improved layout and explanations."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    average_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(12, 10)) 
    plt.plot(recall, precision, color='blue', lw=3,
             label=f'Precision-Recall curve (AP = {average_precision:.2f})')

    thresholds = np.append(thresholds, 0)
    highlight_thresholds = [0.3, 0.5, 0.7]
    annotation_offsets = [(20, -20), (30, 30), (-30, -40)]
    
    for i, threshold in enumerate(highlight_thresholds):
        if threshold < len(thresholds):
            idx = (np.abs(thresholds - threshold)).argmin()
            plt.plot(recall[idx], precision[idx], 'ro', markersize=8)

            offset = annotation_offsets[i % len(annotation_offsets)]
            
            plt.annotate(f'threshold={threshold:.1f}', 
                        (recall[idx], precision[idx]), 
                        xytext=offset,
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                        bbox=dict(boxstyle="round,pad=0.3", alpha=0.1))
    
    plt.xlabel('Recall (True Positive Rate)', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    
    # Add a no-skill baseline
    plt.plot([0, 1], [sum(y_test)/len(y_test)] * 2, linestyle='--', color='grey', 
             label='No skill classifier (AP = {:.2f})'.format(sum(y_test)/len(y_test)))

    explanation = (
        "Precision-Recall Interpretation:\n"
        "• Precision: How many selected items are relevant\n"
        "• Recall: How many relevant items are selected\n"
        "• Higher threshold = higher precision, lower recall\n"
        "• Lower threshold = lower precision, higher recall\n"
        "• Useful for imbalanced datasets"
    )
    
    plt.figtext(0.5, 0.01, explanation, fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.5", alpha=0.1),
               ha='center')
    
    plt.legend(loc="best", fontsize=12)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95]) 
    plt.show()

def plot_all_metrics(clf, X_train, X_test, y_train, y_test):
    """Generate all performance visualizations on a clean grid."""
    plt.style.use('seaborn-v0_8-whitegrid')

    fig = plt.figure(figsize=(20, 16))

    grid_positions = [
        (0, 0, 1, 1),  # row, col, row_span, col_span for Learning Curve
        (0, 1, 1, 1),  # Confusion Matrix
        (1, 0, 1, 1),  # ROC curve
        (1, 1, 1, 1)   # PR curve
    ]
    
    # 1. Learning Curve - top left
    print("\n1. Learning Curve - Shows how model performance improves with more data")
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    _plot_learning_curve_on_ax(clf, X_train, y_train, ax=ax1)
    
    # 2. Confusion Matrix - top right
    print("\n2. Confusion Matrix - Shows prediction accuracy breakdown")
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    
    # Get predictions
    y_pred = clf.predict(X_test)
    
    # Plot confusion matrix
    _plot_confusion_matrix_on_ax(y_test, y_pred, ax=ax2)
    
    # 3. ROC curve - bottom left
    print("\n3. ROC Curve - Shows true positive vs false positive tradeoff")
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    
    # Get probability scores for positive class
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Plot ROC curve
    _plot_roc_curve_on_ax(y_test, y_pred_proba, ax=ax3)
    
    # 4. Precision-Recall curve - bottom right
    print("\n4. Precision-Recall Curve - Shows precision vs recall tradeoff")
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    _plot_pr_curve_on_ax(y_test, y_pred_proba, ax=ax4)

    plt.tight_layout(pad=5.0, h_pad=6.0, w_pad=6.0)

    fig.suptitle('Tweet Classification Model Performance Metrics', 
                 fontsize=20, fontweight='bold', y=0.98)

    plt.subplots_adjust(top=0.92)
    
    plt.show()
    print("\nAll visualizations generated successfully!")

def _plot_learning_curve_on_ax(estimator, X, y, ax=None):
    """Plot learning curve on a specific axis."""
    from sklearn.model_selection import learning_curve
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy')
    
    # Calculate statistics
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot average scores with filled std
    ax.plot(train_sizes, train_mean, 'o-', color='#1f77b4', 
            label='Training score', linewidth=2, markersize=6)
    ax.plot(train_sizes, test_mean, 'o-', color='#ff7f0e', 
            label='Cross-validation score', linewidth=2, markersize=6)
    
    ax.fill_between(train_sizes, train_mean - train_std, 
                    train_mean + train_std, alpha=0.1, color='#1f77b4')
    ax.fill_between(train_sizes, test_mean - test_std, 
                    test_mean + test_std, alpha=0.1, color='#ff7f0e')
    
    # Titles and labels
    ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Examples', fontsize=12)
    ax.set_ylabel('Accuracy Score', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)

    gap = train_mean[-1] - test_mean[-1]
    if gap > 0.05:
        insight = "Model shows overfitting"
    elif test_mean[-1] < 0.6:
        insight = "Model may need more complexity"
    else:
        insight = "Good generalization"

    ax.text(0.05, 0.05, f"{insight}\nGap: {gap:.2f}", 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.1))
    
    ax.grid(True)
    return ax

def _plot_confusion_matrix_on_ax(y_true, y_pred, ax=None, class_names=['Negative', 'Positive']):
    """Plot confusion matrix on a specific axis."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names, ax=ax,
               annot_kws={"size": 14, "weight": "bold"})
    
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    return ax

def _plot_roc_curve_on_ax(y_test, y_pred_proba, ax=None):
    """Plot ROC curve on a specific axis."""
    from sklearn.metrics import roc_curve, auc
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', 
            label='Random classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)

    ax.text(0.7, 0.1, "Higher AUC = Better model\nAUC=1: Perfect model\nAUC=0.5: Random guessing", 
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", alpha=0.1))
    
    ax.grid(True)
    return ax

def _plot_pr_curve_on_ax(y_test, y_pred_proba, ax=None):
    """Plot Precision-Recall curve on a specific axis."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Plot curve
    ax.plot(recall, precision, color='green', lw=2, 
            label=f'PR curve (AP = {avg_precision:.2f})')
    
    # Add no-skill baseline
    ax.plot([0, 1], [sum(y_test)/len(y_test)] * 2, linestyle='--', color='gray', 
            label='No skill')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="best", fontsize=10)
    
    # Add explanation
    ax.text(0.05, 0.2, "Higher AP = Better model\nGood for imbalanced data", 
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", alpha=0.1))
    
    ax.grid(True)
    return ax