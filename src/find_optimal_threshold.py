from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
import numpy as np

def find_optimal_threshold(y_true, y_proba):
    """
    Найти оптимальный threshold для каждого класса
    
    metric: 'f1', 'balanced'
    """

    best_thresholds = np.zeros(y_proba.shape[1])
    best_scores = np.zeros(y_proba.shape[1])

    for i in range(y_proba.shape[1]):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_proba[:, i])
    
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        idx = np.argmax(f1_scores)
        best_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
        best_score = f1_scores[idx]
        
        best_thresholds[i] = best_threshold
        best_scores[i] = best_score

    return best_thresholds.tolist(), best_scores.tolist()

