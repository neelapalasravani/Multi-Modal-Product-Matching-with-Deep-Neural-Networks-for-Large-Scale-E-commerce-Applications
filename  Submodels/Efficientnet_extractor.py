import torch
import timm

class EfficientNetExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0).to(self.device)
        self.model.eval()

    def extract(self, images):
        with torch.no_grad():
            features = self.model(images.to(self.device)).cpu()
        return features.numpy()
