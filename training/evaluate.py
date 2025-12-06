import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import MODEL_EVAL_FIG_DIR, MODEL_EVAL_FIG_PATH, EVAL_HIST_BINS, EVAL_PREDICTION_THRESHOLD, EVAL_FONT_FAMILY

def evaluate_model(model, X_test, y_test, files_test=None):
    print("[*] Evaluating model...")
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > EVAL_PREDICTION_THRESHOLD).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[+] Accuracy: {accuracy:.4f}")
    print("\n[*] Classification report:")
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    target_names = []
    if 0 in unique_labels:
        target_names.append('Benign')
    if 1 in unique_labels:
        target_names.append('Malicious')
    if len(unique_labels) > 1:
        print(classification_report(y_test, y_pred, target_names=target_names, labels=unique_labels))
    else:
        label_name = 'Benign' if unique_labels[0] == 0 else 'Malicious'
        print(f"All samples in test set belong to '{label_name}' category")
        precision = precision_score(y_test, y_pred, zero_division=0)
        print(f"Precision: {precision:.4f}")
    false_positives = []
    if files_test is not None:
        fp_indices = np.where((y_pred == 1) & (y_test == 0))[0]
        false_positives = [files_test[i] for i in fp_indices]
        print(f"\n[*] Detected {len(false_positives)} false positive samples:")
        for fp_file in false_positives[:10]:
            print(f"    - {fp_file}")
        if len(false_positives) > 10:
            print(f"    ... and {len(false_positives) - 10} more false positive samples")
    plt.rcParams['font.sans-serif'] = EVAL_FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = False
    if len(unique_labels) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=target_names, yticklabels=target_names)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        if 0 in unique_labels:
            ax2.hist(y_pred_proba[y_test == 0], bins=EVAL_HIST_BINS, alpha=0.7, label='Benign', color='blue')
        if 1 in unique_labels:
            ax2.hist(y_pred_proba[y_test == 1], bins=EVAL_HIST_BINS, alpha=0.7, label='Malicious', color='red')
        ax2.set_xlabel('Prediction Probability')
        ax2.set_ylabel('Sample Count')
        ax2.set_title('Prediction Probability Distribution')
        ax2.legend()
        plt.tight_layout()
        os.makedirs(MODEL_EVAL_FIG_DIR, exist_ok=True)
        plt.savefig(MODEL_EVAL_FIG_PATH, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"[+] Evaluation charts saved to: {MODEL_EVAL_FIG_PATH}")
    else:
        print("[*] Skipping visualization chart generation as test set contains only one category")
    return accuracy, false_positives
