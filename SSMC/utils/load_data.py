from pathlib import Path
import torch

def load_data(config, encoder, split, target_section=None):
    '''
    Loads saved embeddings and the text files associated with them for specified sections
    '''
    root = Path(__file__).resolve().parent.parent
    results = {}

    dir_split = "val" if split == "validation" else split
    embedding_dir = root / "data" / "embeddings" / f"{encoder}-{dir_split}"
    
    if target_section:
        sections = [target_section]
    else:
        sections = config['data'][split].keys()
    
    for section in sections:
        embedding_file = f"{section}.pt"
        embedding_path = embedding_dir / embedding_file
        
        if not embedding_path.exists():
            raise ValueError(f"Embedding file not found: {embedding_path}")
        
        data = torch.load(embedding_path)
        embeddings = data['embeddings']
        memo_paths = data['memo_paths']
        texts = []
        
        if 'texts' in data:
            texts = data['texts']
        else:
            path = root / config['data'][split][section]
            try:
                for m in memo_paths:
                    with open(path / m, "r", encoding="utf-8") as f:
                        texts.append(f.read().strip())
            except FileNotFoundError:
                print(f"Warning: Text files not found for {section}. Using empty texts.")
                texts = [""] * len(memo_paths)
        
        results[section] = (embeddings, texts)
    
    return results