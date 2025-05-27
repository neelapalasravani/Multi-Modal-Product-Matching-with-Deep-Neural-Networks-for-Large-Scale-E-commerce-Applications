import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate(true_labels, predicted_labels):
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    return f1, precision, recall

if __name__ == "__main__":
    import numpy as np
    # Load ground truth labels
    train_df = pd.read_csv("/kaggle/input/shopee-product-matching/train.csv")
    true_labels = train_df["label_group"].values

    # Load predictions (this should be your final_preds_for_train list or array)
    predicted_labels = np.load("outputs/final_preds_train.npy")

    f1, precision, recall = evaluate(true_labels, predicted_labels)
    print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
