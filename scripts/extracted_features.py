import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.nfnet_extractor import NFNetExtractor
from models.swin_transformer import SwinTransformerExtractor
from models.efficientnet_extractor import EfficientNetExtractor
from models.distilbert_extractor import DistilBertExtractor
from models.albert_extractor import AlbertExtractor
from models.multilingual_bert import MultilingualBertExtractor
from models.tfidf_extractor import TFIDFExtractor
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def batch_extract_features(model, image_paths, batch_size=32):
    features_list = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Extracting {model.__class__.__name__}"):
        batch = image_paths[i:i + batch_size]
        batch_images = [image_transform(Image.open(img).convert("RGB")) for img in batch]
        batch_images = torch.stack(batch_images).to(device)
        features = model.extract(batch_images)
        features_list.append(features)
    return np.vstack(features_list)

def main():
    data_dir = "/kaggle/input/shopee-product-matching"
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train_images_dir = os.path.join(data_dir, "train_images")
    test_images_dir = os.path.join(data_dir, "test_images")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_images = [os.path.join(train_images_dir, img) for img in train_df["image"]]
    test_images = [os.path.join(test_images_dir, img) for img in test_df["image"]]

    nfnet_model = NFNetExtractor().to(device)
    swin_model = SwinTransformerExtractor().to(device)
    efficientnet_model = EfficientNetExtractor().to(device)

    distilbert_model = DistilBertExtractor().to(device)
    albert_model = AlbertExtractor().to(device)
    multilingual_bert_model = MultilingualBertExtractor().to(device)

    tfidf_model = TFIDFExtractor(train_df["title"].tolist())

    # Extract image features
    nfnet_train_features = batch_extract_features(nfnet_model, train_images)
    swin_train_features = batch_extract_features(swin_model, train_images)
    efficientnet_train_features = batch_extract_features(efficientnet_model, train_images)

    nfnet_test_features = batch_extract_features(nfnet_model, test_images)
    swin_test_features = batch_extract_features(swin_model, test_images)
    efficientnet_test_features = batch_extract_features(efficientnet_model, test_images)

    # Save image features
    np.save("outputs/nfnet_features_train.npy", nfnet_train_features)
    np.save("outputs/swin_features_train.npy", swin_train_features)
    np.save("outputs/efficientnet_features_train.npy", efficientnet_train_features)

    np.save("outputs/nfnet_features_test.npy", nfnet_test_features)
    np.save("outputs/swin_features_test.npy", swin_test_features)
    np.save("outputs/efficientnet_features_test.npy", efficientnet_test_features)

    # Text feature extraction with multiprocessing can be added here similarly
    # Or batch processing (not included here for brevity)

    print("🚀 Feature extraction completed!")

if __name__ == "__main__":
    main()
