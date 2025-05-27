import numpy as np
import faiss

def build_faiss_index(features_path, use_gpu=True):
    features = np.load(features_path)
    index = faiss.IndexFlatL2(features.shape[1])  # L2 distance index

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(features)
    return index

if __name__ == "__main__":
    index = build_faiss_index("outputs/combined_features.npy")
    print("✅ FAISS index built and loaded.")
