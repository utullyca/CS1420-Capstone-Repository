import os
import yaml
import torch
import gc
import time
from tqdm import tqdm
from pathlib import Path
from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

def clear_gpu_memory(): # Same Memory management from few-shot
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)

def load_config(): # Same config loading from everything
    config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_model(config): 
    '''
    Sets up the mistral from the config
    '''
    print(f"Loading model: {config['model']['name']}")
    
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(config["model"].get("dtype", "float16"), torch.float16)
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        device_map=config["model"].get("device_map", "auto"),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN") if config["model"].get("use_auth_token", False) else None,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN") if config["model"].get("use_auth_token", False) else None,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_data(section_path):
    '''
    Gets all the annotated training data for a specific section and creates a list
    '''
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / section_path
    
    print(f"Loading data from {data_dir}")
    
    data = []
    files = list(data_dir.glob("*.txt"))
    
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        if "######" in content:
            text, feedback = content.split("######", 1)
            data.append({
                "text": text.strip(),
                "feedback": feedback.strip(),
                "file": file_path.name
            })
    
    return data

def embed_expertise(example, section):
    '''
    embed the expertise from the annotated data so mistral can learn key patterns
    '''
    text = example["text"]
    feedback = example["feedback"]
    prompt = f"<s>[INST] Here is a legal writing {section} example:\n\n{text}\n\nProvide detailed feedback on this {section} example. [/INST]\n\n{feedback}</s>"
    return prompt

def save_checkpoint(model, output_dir, epoch=None, is_best=False):
    '''
    save the best model
    '''
    if epoch is not None:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = checkpoint_dir / f"epoch_{epoch+1}"
    elif is_best:
        save_path = output_dir / "best_model"
    else:
        save_path = output_dir / "checkpoint"
    
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    
    return save_path


def train(model, tokenizer, train_data, optimizer, section, epoch, num_epochs):
    '''
    fine-tuning mistral w/ 12/15 annotated examples I had
    '''
    model.train()
    loss = 0
    # tqdm is pretty fun to stare at
    for i in tqdm(range(len(train_data)), desc=f"Epoch {epoch+1}/{num_epochs}"):
        example = train_data[i] # gets current example to train on
        
        text = embed_expertise(example, section) # encodes the expertise
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        input_ids = tokens["input_ids"].to(model.device)
        attention_mask = tokens["attention_mask"].to(model.device)
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        batch_loss = outputs.loss
        batch_loss.backward()
        # small dataset so gradient clipping to stop any large updates (stabilize learning)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss += batch_loss.item()
    
    avg_loss = loss / len(train_data) if train_data else 0
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    
    return avg_loss

def validate(model, tokenizer, val_data, section):
    '''
    valideating every epoch with all 3 validation examples
    '''
    model.eval()
    vloss = 0
    
    with torch.no_grad():
        for example in tqdm(val_data, desc="Validation"):
            text = embed_expertise(example, section)
            tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            
            input_ids = tokens["input_ids"].to(model.device)
            attention_mask = tokens["attention_mask"].to(model.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            
            vloss += outputs.loss.item()
    avg_vloss = vloss / len(val_data) if val_data else 0
    # Thought about using but never saved it along with the model so it
    # wasn't used
    pplx = torch.exp(torch.tensor(avg_vloss)).item()
    print(f"Validation Perplexity: {pplx:.4f}")
    return pplx


def main():
    import sys    
    config = load_config()
    project_dir = Path(__file__).resolve().parent.parent
    output_dir = project_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)    
    model, tokenizer = load_model(config)
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    sections = list(config["data"]["evals"].keys())
    
    # train all the sections
    for section in sections:
        print(f"Training section: {section}")
        train_data = load_data(config["data"]["evals"][section])
        val_data = load_data(config["data"]["validation"][section])
        
        section_output = output_dir / section
        section_output.mkdir(parents=True, exist_ok=True)
        
        epochs = int(config["training"]["num_train_epochs"])
        lr = float(config["training"]["learning_rate"])
        weight_decay = float(config["training"]["weight_decay"])
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # validated every epoch
        best_pplx = float('inf')
        for epoch in range(epochs):
            train(model, tokenizer, train_data, optimizer, section, epoch, epochs)
            pplx = validate(model, tokenizer, val_data, section)
            save_checkpoint(model, section_output, epoch=epoch)
            if pplx < best_pplx:
                best_pplx = pplx
                print(f"New best perplexity: {best_pplx:.4f}")
                save_checkpoint(model, section_output, is_best=True)
        clear_gpu_memory()
        print(f"Completed training for section: {section}")

if __name__ == "__main__":
    main()