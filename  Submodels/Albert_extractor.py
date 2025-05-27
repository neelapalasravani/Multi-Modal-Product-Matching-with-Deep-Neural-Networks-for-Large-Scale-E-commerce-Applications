import torch
from transformers import AlbertTokenizer, AlbertModel

class AlbertExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.model = AlbertModel.from_pretrained('albert-base-v2').to(self.device)
        self.model.eval()

    def extract(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
        with torch.no_grad():
            output = self.model(**{k: v.to(self.device) for k, v in tokens.items()})
            features = output.last_hidden_state.mean(dim=1).cpu().numpy()
        return features
