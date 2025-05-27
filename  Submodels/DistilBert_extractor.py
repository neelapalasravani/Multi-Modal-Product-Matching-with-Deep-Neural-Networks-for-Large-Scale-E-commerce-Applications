import torch
from transformers import DistilBertTokenizer, DistilBertModel

class DistilBertExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        self.model.eval()

    def extract(self, texts):
        """
        texts: list of strings
        Returns numpy array of shape (len(texts), hidden_dim)
        """
        tokens = self.tokenizer(texts, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
        with torch.no_grad():
            output = self.model(**{k: v.to(self.device) for k, v in tokens.items()})
            features = output.last_hidden_state.mean(dim=1).cpu().numpy()
        return features
