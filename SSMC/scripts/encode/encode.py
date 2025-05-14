import torch
from pathlib import Path
import yaml
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from model.bert_encoder import BERTEncoder
from model.legal_bert_encoder import LegalBERTEncoder

def load_config():
    config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def encode_data(input_path, output_path, encoder):
    '''
    batch encodes data using bert or legal_bert
    '''
    path = Path(input_path)
    memos = []
    memo_paths = []

    # encodes all the data, split by encoder and train or val in the main call and saves in different dirs
    for memo_path in path.glob("*.txt"):
        with open(memo_path, "r", encoding="utf-8") as f:
            memo = f.read().strip()
            memos.append(memo)
            memo_paths.append(str(memo_path.name))
    if not memos:
        raise ValueError("No memos found in input path")
    
    print(f"Encoding {len(memos)} memos from {input_path}")
    embeddings = encoder.encode(memos)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'embeddings': embeddings,
        'memo_paths': memo_paths
    }, output_path)

    print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    config = load_config()
    root = Path(__file__).resolve().parent.parent.parent
    output_base = root / "data" / "embeddings"
    output_base.mkdir(parents=True, exist_ok=True)
    
    bert_train_dir = output_base / "bert-train"
    bert_val_dir = output_base / "bert-val"
    legal_bert_train_dir = output_base / "legal-bert-train"
    legal_bert_val_dir = output_base / "legal-bert-val"
    
    for dir_path in [bert_train_dir, bert_val_dir, legal_bert_train_dir, legal_bert_val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    train = {section: root / path for section, path in config['data']['train'].items()}
    val = {section: root / path for section, path in config['data']['validation'].items()}

    bert = BERTEncoder()
    legal_bert = LegalBERTEncoder()

    for section, path in train.items():
        encode_data(path, bert_train_dir / f"{section}.pt", bert)
        encode_data(path, legal_bert_train_dir / f"{section}.pt", legal_bert)
    
    for section, path in val.items():
        encode_data(path, bert_val_dir / f"{section}.pt", bert)
        encode_data(path, legal_bert_val_dir / f"{section}.pt", legal_bert)