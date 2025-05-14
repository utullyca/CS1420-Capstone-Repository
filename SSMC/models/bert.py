import torch
from transformers import BertModel, BertTokenizer

# https://stackoverflow.com/questions/69517460/bert-get-sentence-embedding
'''
Bert encoder thanks to stack
tweaked the code to encode tokens
I don't fully understand what's happening but i get it's the creation of the dense embedding vector for SSMC
'''
class BERTEncoder:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def encode(self, text, batch_size=5):
        embeddings = []
        for i in range(0, len(text), batch_size):
            batch = text[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0] # using CLS token to capture max nuance
            embeddings.append(batch_embeddings)
        return torch.concat(embeddings, dim=0)