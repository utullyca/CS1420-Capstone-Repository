import os
import yaml
import torch
import gc
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda")
print(f"Using CUDA - Available GPUs: {torch.cuda.device_count()}")

def clear_gpu_memory(): # Yay for memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1) 

def load_config():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()
model_id = config['model']['name']

def load_fine_tuned(section_path):    
    '''
    loads the fine-tuned model for a specifc IREAC section and sets it up
    '''
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=config['model']['trust_remote_code'],
        use_auth_token=config['model']['use_auth_token']
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=config['model']['use_auth_token'])
    tokenizer.pad_token = tokenizer.eos_token
    
    model_path = section_path / "best_model"
    if model_path.exists():
        print(f"Loading best model from {model_path}")
        ft_model = PeftModel.from_pretrained(model, model_path)
    return ft_model, tokenizer

def embed_expertise(text, section, tokenizer):
    '''
    Sets up mistral conversation format (system <-> user) with the system prompt
    (Making Mistral believe its an expert LRW prof.)
    '''
    section_name = {
        "issue_statements": "issue statement",
        "rule_statements": "rule statement",
        "case_comparisons": "case comparison",
        "rule_applications": "rule application",
        "writing": "overall legal writing"
    }.get(section, section)
    
    prompt = f"You are an expert legal writing instructor providing feedback on student {section_name}s."
    
    message = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Please provide feedback on this legal writing:\n\n{text}"}
    ]
    
    return tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

def evaluate(section, text):
    '''
    generates feedback using section-specific fine-tuned mistral model
    '''
    try:
        model_path = Path(__file__).parent.parent / "outputs" / section
        
        model, tokenizer = load_fine_tuned(model_path)
        inputs = embed_expertise(text, section, tokenizer)
        print(f"Generating feedback for {section}")
        
        outputs = model.generate(
            inputs,
            **config['generation']
        )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated
    except Exception as e:
        print(f"Error evaluating {section}: {str(e)}")
        return None
    finally:
        clear_gpu_memory()

def read_files(section):
    '''
    gets file from the validation folder to run the evaluation on
    fundamentally same as the few-shot load_examples
    '''
    data_path = Path(__file__).parent.parent / config['data']['validation'][section]
    files = list(data_path.glob("*.txt"))
    examples = []
    for file in files:
        try:
            with open(file, "r") as f:
                content = f.read()
                parts = content.split("######")  
                if len(parts) != 2:
                    print(f"Invalid format in {file}")
                    continue
                text = parts[0].strip()
                feedback = parts[1].strip()
                examples.append({"text": text, "feedback": feedback, "file_name": file.stem})
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    return examples

if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    sections = ["issue_statements", "rule_statements", "case_comparisons", "rule_applications", "writing"]
    
    # runs evaluation for all sections
    for section in sections:
        print(f"\nEvaluating {section}")
        section_dir = output_dir / section
        section_dir.mkdir(exist_ok=True)
        
        # Runs evaluate on all 3/15 holdouts (only passes in the text)
        examples = read_files(section)            
        for example in examples:
            result = evaluate(section, example["text"])
            if result:
                with open(section_dir / f"{example['file_name']}.txt", "w") as f:
                    f.write(result)
        clear_gpu_memory()
    print("All evaluations complete!")