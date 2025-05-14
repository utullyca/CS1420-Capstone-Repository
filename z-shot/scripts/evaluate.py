from pathlib import Path
from typing import List, Dict
import os
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA - Available GPUs: {torch.cuda.device_count()}")
else:
    device = torch.device("cpu")
    print("Using CPU")

def load_config() -> Dict:
    """
    Loads configuration from YAML file.
    """
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
model_id = config['model']['name']
dtype = getattr(torch, config['model'].get('dtype', 'float32'))

print(f"Loading model {model_id} with dtype {dtype}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float32,  # full precision
    use_auth_token=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)

'''
Note no data exists in the repository. This is strictly for submission.
'''
def evaluate_issue_statement_prompt(issue_statement: str) -> List[Dict[str, str]]:
    """
    Load and format the prompt template for z-shot issue statement evaluation.
    
    This method loads a predefined prompt template that guides the model to evaluate
    legal issue statements according to specific criteria including format, content,
    length, clarity, legal accuracy, and factual presentation.
    
    Args:
        issue_statement (str): the issue statement to be evaluated.
    """
    prompt = Path(config['prompts']['issue_prompt_template'])
    with open(prompt, "r") as file:
        evaluation_guide = file.read()
    
    return [{
        "role": "system",
        "content": evaluation_guide
    },
    {
        "role": "user",
        "content": f"The issue statement to review is:\n\n{issue_statement}"
    }]

def evaluate_issue_statement(issue_statement: str) -> str:
    """
    Perform zero-shot evaluation of a legal memo issue statement using Mistral-7B-Instruct.
    
    Args:
        issue_statement (str): The student's issue statement to evaluate.
    
    Returns:
        str: The model's evaluation based on the return from @evaluate_issue_statement_prompt.
    """
    try:
        issue_prompt = evaluate_issue_statement_prompt(issue_statement)
        inputs = tokenizer.apply_chat_template(
            issue_prompt, 
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        
        outputs = model.generate(
            **inputs,
            **config['generation']
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    except Exception as e:
        return f"Error: {str(e)}"

def evaluate_rule_statement_prompt(rule_statement: str) -> List[Dict[str, str]]:
    """
    Load and format the prompt template for z-shot rule statement evaluation.
    
    This method loads a predefined prompt template that guides the model to evaluate
    legal rule statements according to specific criteria including generality,
    test identification, clarity, synthesis, and proper citation.
    
    Args:
        rule_statement (str): the rule statement to be evaluated.
    """
    prompt = Path(config['prompts']['rule_prompt_template'])
    with open(prompt, "r") as file:
        evaluation_guide = file.read()
    
    return [{
        "role": "system",
        "content": evaluation_guide
    },
    {
        "role": "user",
        "content": f"The rule statement to review is:\n\n{rule_statement}"
    }]

def evaluate_rule_statement(rule_statement: str) -> str:
    """
    Perform zero-shot evaluation of a legal memo rule statement using Mistral-7B-Instruct.
    
    Args:
        rule_statement (str): The student's rule statement to evaluate.
    
    Returns:
        str: The model's evaluation based on the return from @evaluate_rule_statement_prompt.
    """
    try:
        rule_prompt = evaluate_rule_statement_prompt(rule_statement)
        inputs = tokenizer.apply_chat_template(
            rule_prompt, 
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        
        outputs = model.generate(
            **inputs,
            **config['generation']
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    except Exception as e:
        return f"Error: {str(e)}"


def evaluate_case_comparison_prompt(case_comparison: str) -> List[Dict[str, str]]:
    """
    Load and format the prompt template for z-shot case comparison (rule explanation) evaluation.
    
    This method loads a predefined prompt template that guides the model to evaluate
    legal case comparisons according to specific criteria including case selection,
    explanation depth, transitions, and proper structuring.
    
    Args:
        case_comparison (str): the case comparison to be evaluated.
    """
    prompt = Path(config['prompts']['case_comparison_prompt_template'])
    with open(prompt, "r") as file:
        evaluation_guide = file.read()
    
    return [{
        "role": "system",
        "content": evaluation_guide
    },
    {
        "role": "user",
        "content": f"The case comparison to review is:\n\n{case_comparison}"
    }]

def evaluate_case_comparison(case_comparison: str) -> str:
    """
    Perform zero-shot evaluation of a legal memo case comparison/rule explanation using Mistral-7B-Instruct.
    
    Args:
        case_comparison (str): The student's case comparison to evaluate.
    
    Returns:
        str: The model's evaluation based on the return from @evaluate_case_comparison_prompt.
    """
    try:
        cc_prompt = evaluate_case_comparison_prompt(case_comparison)
        inputs = tokenizer.apply_chat_template(
            cc_prompt, 
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        
        outputs = model.generate(
            **inputs,
            **config['generation']
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    except Exception as e:
        return f"Error: {str(e)}"


def evaluate_rule_application_prompt(rule_application: str) -> List[Dict[str, str]]:
    """
    Load and format the prompt template for z-shot rule application evaluation.
    
    This method loads a predefined prompt template that guides the model to evaluate
    legal rule applications according to specific criteria including analogical reasoning,
    fact comparison, and handling of favorable and unfavorable precedents.
    
    Args:
        rule_application (str): the rule application to be evaluated.
    """
    prompt = Path(config['prompts']['rule_application_prompt_template'])
    with open(prompt, "r") as file:
        evaluation_guide = file.read()
    
    return [{
        "role": "system",
        "content": evaluation_guide
    },
    {
        "role": "user",
        "content": f"The rule application to review is:\n\n{rule_application}"
    }]

def evaluate_rule_application(rule_application: str) -> str:
    """
    Perform zero-shot evaluation of a legal memo rule application using Mistral-7B-Instruct.
    
    Args:
        rule_application (str): The student's rule application to evaluate.
    
    Returns:
        str: The model's evaluation based on the return from @evaluate_rule_application_prompt.
    """
    try:
        ra_prompt = evaluate_rule_application_prompt(rule_application)
        inputs = tokenizer.apply_chat_template(
            ra_prompt, 
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        
        outputs = model.generate(
            **inputs,
            **config['generation']
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    except Exception as e:
        return f"Error: {str(e)}"