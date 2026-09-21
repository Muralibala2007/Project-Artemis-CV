import json
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report

def diagnose():
    df = pd.read_csv('oof_predictions.csv')
    with open('best_threshold.json') as f:
        threshold = json.load(f)['threshold']

    probs = df['oof_prob_class1'].values
    labels = df['label'].values
    preds = (probs >= threshold).astype(int)

    print("=== Out-of-fold predictions across ALL training samples (unbiased) ===")
    print(f"n = {len(labels)}  |  threshold used: {threshold:.2f}")
    print(f"True label distribution:  0: {(labels==0).sum()}, 1: {(labels==1).sum()}")
    print(f"Pred label distribution:  0: {(preds==0).sum()}, 1: {(preds==1).sum()}")
    print()
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(labels, preds))
    print()
    print(classification_report(labels, preds, target_names=["Depth(0)", "Rise(1)"]))
    print(f"Balanced accuracy: {balanced_accuracy_score(labels, preds):.4f}")

if __name__ == '__main__':
    diagnose()