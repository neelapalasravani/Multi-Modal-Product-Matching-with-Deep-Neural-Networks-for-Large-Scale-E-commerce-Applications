import torch
import timm

class SwinTransformerExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0).to(self.device)
        self.model.eval()

    def extract(self, images):
        """
        Extract features from a batch of images.
        images: torch.Tensor
        """
        with torch.no_grad():
            features = self.model(images.to(self.device)).cpu()
        return features.numpy()
