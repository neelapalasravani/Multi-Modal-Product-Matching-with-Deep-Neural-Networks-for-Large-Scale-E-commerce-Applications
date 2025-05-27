import numpy as np

def get_similar_products(index, query_features, top_k=5):
    distances, indices = index.search(query_features, top_k)
    return indices

def batch_search(index, features, top_k=5, batch_size=32):
    results = []
    for i in range(0, len(features), batch_size):
        batch_features = features[i:i + batch_size]
        batch_results = get_similar_products(index, batch_features, top_k)
        results.extend(batch_results)
    return results

if __name__ == "__main__":
    import faiss
    features = np.load("outputs/combined_features.npy")
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)

    predictions = batch_search(index, features, top_k=5)
    print("✅ Similarity search completed.")
