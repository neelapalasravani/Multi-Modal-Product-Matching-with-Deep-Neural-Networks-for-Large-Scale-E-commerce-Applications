import torch
from transformers import AutoTokenizer, AutoModel

class MultilingualBertExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = AutoModel.from_pretrained('bert-base-multilingual-cased').to(self.device)
        self.model.eval()

    def extract(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
        with torch.no_grad():
            output = self.model(**{k: v.to(self.device) for k, v in tokens.items()})
            features = output.last_hidden_state.mean(dim=1).cpu().numpy()
        return features
