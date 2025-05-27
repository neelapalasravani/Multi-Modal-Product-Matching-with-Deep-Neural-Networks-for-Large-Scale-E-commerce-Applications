# Multi-Modal-Product-Matching-with-Deep-Neural-Networks-for-Large-Scale-E-commerce-Applications

## Project Overview

Online e-commerce platforms face a critical challenge of **duplicate product listings** that confuse buyers, mislead pricing, and reduce trust in the platform. Sellers also suffer from copycat listings and price manipulations. Traditional matching methods struggle due to visual and textual variations in product images and descriptions.

This project develops a **multi-modal deep learning system** that combines image and text embeddings to accurately detect duplicate product listings at scale. The solution improves search relevance, reduces fraud, and helps clean up product catalogs on large platforms like Shopee.

Dataset Link: https://www.kaggle.com/competitions/shopee-product-matching/data
---

```
 

Multi-Modal-Product-Matching/
│
├── README.md                      # Project overview + usage + results summary
├── requirements.txt               
│
├── data/                         # Dataset info + README 
│   └── README.md                 
│
├── notebooks/                    # Jupyter notebooks 
│   ├── feature_extraction_7submodels.ipynb   # Your 7 submodels extraction & fusion
│   ├── feature_extraction_2models.ipynb      # Simplified 2 models notebook
│   └── analysis.ipynb            # Evaluation, visualization, result analysis
│
├── models/                      # Modular model code 
│   ├── nfnet_extractor.py        # NFNet feature extraction class
│   ├── swin_transformer.py       # Swin Transformer extractor class
│   ├── efficientnet_extractor.py # EfficientNet extractor class
│   ├── distilbert_extractor.py   # DistilBERT text extractor class
│   ├── albert_extractor.py       # ALBERT extractor class
│   ├── multilingual_bert.py      # Multilingual BERT extractor class
│   ├── tfidf_extractor.py        # TF-IDF extractor class
│   └── feature_fusion.py         # Combine features
│
├── scripts/                    # Scripts for running training, inference, postprocessing
│   ├── extract_features.py       # Runs all extraction pipeline, saves .npy files
│   ├── build_faiss_index.py      # Loads features, builds FAISS index
│   ├── run_similarity_search.py  # Query FAISS, get top-k neighbors
│   ├── voting.py                 # Implements voting mechanism & final predictions
│   ├── evaluate.py               # Computes F1, precision, recall etc.
│   └── postprocess.py            # Optional cleaning & thresholding                 
│
├── utils/                      # Utility functions 
│   └── utils.py
│
└── docs/                       # Final Work in doc
    ├── Final_Project_Report.pdf
    └── Poster_Presentation.pdf
```

## Why This Project?

- Duplicate product listings confuse buyers and mislead pricing.
- Sellers face negative effects from copycat listings and price manipulation.
- Visual and textual differences make traditional matching methods ineffective.
- Our system integrates deep learning on images and text to detect duplicates.
- This helps improve search, reduce fraud, and clean product catalogs efficiently.

---

## Objectives

- Understand the multimodal challenges in product matching.
- Develop and compare multiple state-of-the-art deep learning models.
- Evaluate models using **F1 Score**, **Precision**, and **Recall**.
- Ensure scalability and real-world applicability for large datasets.

---

## Dataset

- **Source:** Shopee Product Matching Dataset (Kaggle)
- **Size:** 34,251 product listings and 32,412 associated images
- **Key Features:** `posting_id`, `image`, `title`, `label_group`
- **Challenges:** Multilingual noise, class imbalance, and scalability concerns

---

## Models Explored

### Model 1
- **Architecture:** ECA NFNet L1 + Paraphrase-XLM-R + FAISS + INB
- **Image Model:** Efficient Channel Attention + Normalizer-Free ResNet (NFNet)
- **Text Model:** Paraphrase-XLM-R Multilingual embedding

### Model 2
- **Architecture:** NFNet + Swin Transformer + EfficientNet + Distil-BERT + ALBERT + Multilingual BERT + KNN Voting
- **Image Models:** NFNet, Swin Transformer, EfficientNet
- **Text Models:** Distil-BERT, ALBERT, Multilingual BERT, TF-IDF

### Model 3 (Planned)
- **Architecture:** ViT + NFNet-F0 + Indonesian-BERT + Multilingual-BERT + Paraphrase-XLM + GAT
- **Image Models:** Vision Transformer (ViT), NFNet-F0
- **Text Models:** Indonesian-BERT, Multilingual-BERT, Paraphrase-XLM

---

**RESULTS**

| Model   | F1 Score | Precision | Recall  |
|---------|----------|-----------|---------|
| Model 1 | 0.9097   | 0.8657    | 0.8731  |
| Model 2 | 0.14     | 0.554     | 0.8135  |
| Model 3 | 0.5324   | 0.4271    | 0.8858  |

Model 1 achieved the highest performance across all metrics and is considered the best for this task.

---

## How to Use This Repository

1. Explore the `notebooks/` folder to review model implementations.
2. Use the provided scripts in the `models/` folder for training and evaluation.
3. Dataset details and download instructions are provided in the `data/` folder.
4. Run experiments and evaluate models using the provided notebooks.


Feel free to reach out for questions or contributions!

