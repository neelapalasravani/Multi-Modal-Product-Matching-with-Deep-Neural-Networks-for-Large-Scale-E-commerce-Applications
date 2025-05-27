import os
import numpy as np
import pandas as pd
import faiss
from sklearn.metrics import f1_score

from models.nfnet_extractor import NFNetExtractor
from models.swin_transformer import SwinTransformerExtractor
from models.efficientnet_extractor import EfficientNetExtractor
from models.distilbert_extractor import DistilBertExtractor
from models.albert_extractor import AlbertExtractor
from models.multilingual_bert import MultilingualBertExtractor
from models.tfidf_extractor import TFIDFExtractor

from scripts.voting import final_prediction
from scripts.postprocess import post_process

def load_features():
    print("Loading features...")
    nfnet_train = np.load("outputs/nfnet_features_train.npy")
    swin_train = np.load("outputs/swin_features_train.npy")
    efficientnet_train = np.load("outputs/efficientnet_features_train.npy")
    combined = np.hstack([nfnet_train, swin_train, efficientnet_train])
    return combined

def build_faiss_index(features):
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(features.shape[1])
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(features)
    return index

def search_similar(index, features, top_k=5):
    print(f"Running similarity search for top-{top_k} neighbors...")
    batch_size = 32
    predictions = []
    for i in range(0, len(features), batch_size):
        batch = features[i:i+batch_size]
        _, inds = index.search(batch, top_k)
        predictions.extend(inds)
    return predictions

def load_labels():
    print("Loading true labels...")
    df = pd.read_csv("/kaggle/input/shopee-product-matching/train.csv")
    return df["label_group"].values

def map_predictions_to_labels(preds, label_map):
    mapped = []
    for pred in preds:
        if pred in label_map:
            mapped.append(label_map[pred])
        else:
            mapped.append(-1)  # or some default label
    return mapped

def main():
    features = load_features()
    index = build_faiss_index(features)
    preds = search_similar(index, features)

    # Voting to get final predicted label indices
    final_preds = final_prediction(preds, threshold=0.5)

    # Load true labels
    true_labels = load_labels()

    # Create mapping from prediction index to true label
    unique_labels = np.unique(true_labels)
    label_map = {i: label for i, label in enumerate(unique_labels)}

    # Map predictions to true label space
    final_preds_mapped = map_predictions_to_labels(final_preds, label_map)

    # Post-process predictions (e.g., fix unmatched)
    final_preds_processed = post_process(final_preds_mapped)

    # Evaluate
    f1 = f1_score(true_labels, final_preds_processed, average='macro')
    print(f"Final Macro F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
