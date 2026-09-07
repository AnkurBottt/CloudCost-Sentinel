import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def evaluate_model(y_true, probabilities, threshold=0.5):
    predictions = (np.array(probabilities) >= threshold).astype(int)

    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "anomaly_rate": np.mean(y_true),
        "predicted_anomaly_rate": np.mean(predictions),
        "confusion_matrix": confusion_matrix(y_true, predictions),
    }


def print_metrics(metrics, model_name):
    print(f"\n{model_name}")
    print("-" * len(model_name))
    print(f"Threshold : {metrics['threshold']:.2f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"Actual anomaly rate    : {metrics['anomaly_rate']:.2%}")
    print(f"Predicted anomaly rate : {metrics['predicted_anomaly_rate']:.2%}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])


def find_best_threshold(y_true, probabilities):
    # old search was only 0.3-0.7; for an imbalanced detector useful thresholds can be lower
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:
        metrics = evaluate_model(y_true, probabilities, threshold)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold

    return best_threshold
