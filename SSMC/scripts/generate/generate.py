from models.frozen_mistral import FrozenMistral
from models.mlpc import MLPC
from models.bert import BERTEncoder
from models.legal_bert import LegalBERTEncoder


import sys
import os
import importlib
import torch
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def load_config():
    config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def init_model(section, encoder, device):
    '''
    init the model with the correct endcoder and saved weight
    '''
    if encoder == "bert":
        encoder_model = BERTEncoder()
        encoder_name = "bert"
    elif encoder == "legal_bert" or encoder == "legal-bert":
        encoder_model = LegalBERTEncoder()
        encoder_name = "legal-bert"  
    else:
        raise ValueError(f"Unknown encoder: {encoder}")
        
    saved_models = Path(__file__).resolve().parent.parent.parent / "saved_models" / "a6000"
    model_path = saved_models / f"mlpc_{encoder_name}_{section}.pt"
    
    # load the saved model
    mlpc = MLPC().to(device)
    mlpc.load_state_dict(torch.load(model_path, map_location=device))
    mlpc.eval()

    # have to manually set device if not using ddp 
    if device.type == 'cuda':
        device_for_mistral = torch.device('cuda:0')
    else:
        device_for_mistral = device
    
    frozen_mistral = FrozenMistral(device_for_mistral)
    
    return mlpc, encoder_model, frozen_mistral

def load_data(section):
    '''
    get a random piece of validation data to be used as the example for generation
    '''
    config = load_config()
    section_path = Path(__file__).resolve().parent.parent.parent / config['data']['validation'][section]
    data = list(section_path.glob("*.txt"))
    file = np.random.choice(data)
    with open(file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text

def generate_data(section, encoder, device):
    try:
        mlpc, encoder_model, frozen_mistral = init_model(section, encoder, device)
        input_text = load_data(section)
        embedding = encoder_model.encode([input_text])
        embedding = embedding.to(device)
        print(f"Generating soft prompt")
        with torch.no_grad():
            soft_prompt = mlpc(embedding)
            soft_prompt = soft_prompt.half()
        print(f"Generating output with frozen_mistral")
        output = frozen_mistral.generate(soft_prompt, section, input_text=input_text)
        
        print(f"Decoding output")
        decoded = frozen_mistral.decode(output[0])
        
        # Additional check to ensure we're not returning prompt content (this happend to me a bunch)
        if not decoded.strip():
            # Try again with a different seed but maintain the same temperature - forced regeneration
            output = frozen_mistral.generate(soft_prompt, section, input_text=input_text)
            decoded = frozen_mistral.decode(output[0])
        
        return decoded
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    sections = ["issue_statements", "rule_statements", "case_comparisons", "rule_applications"]
    encoders = ["bert", "legal_bert"]  
    num_samples = 5  

    output_dir = Path(__file__).resolve().parent.parent.parent / "synthetic_data"
    output_dir.mkdir(exist_ok=True, parents=True)

    saved_models_dir = Path(__file__).resolve().parent.parent.parent / "saved_models" / "a6000"
    if saved_models_dir.exists():
        for model_file in saved_models_dir.glob("*.pt"):
            print(f"  {model_file.name}")
    
    # generate data for all 8 section/encoder combos
    for section in sections:
        for encoder in encoders:
            print(f"Generating {num_samples} samples for {section} with {encoder}")
            
            combo_dir = output_dir / f"{section}_{encoder.replace('_', '-')}"
            combo_dir.mkdir(exist_ok=True, parents=True)
            
            for i in range(num_samples):
                try:
                    print(f"  Sample {i+1}/{num_samples}")
                    generated = generate_data(section, encoder, device)
                    output_file = combo_dir / f"sample_{i+1}.txt"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(generated)
                    print(f"  Success: {output_file}")
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
            print(f"Completed generation for {section} with {encoder}")

if __name__ == "__main__":
    main()