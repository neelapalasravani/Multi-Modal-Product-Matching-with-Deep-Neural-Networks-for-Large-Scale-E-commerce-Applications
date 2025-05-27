import torch
import timm

class NFNetExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = timm.create_model('nfnet_f0', pretrained=True, num_classes=0).to(self.device)
        self.model.eval()

    def extract(self, images):
        """
        Extract features from a batch of images.
        images: torch.Tensor of shape (batch_size, 3, H, W)
        Returns numpy array of features.
        """
        with torch.no_grad():
            features = self.model(images.to(self.device)).cpu()
        return features.numpy()
