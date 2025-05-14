from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml
from pathlib import Path
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Running on GPU
device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.device_count()} available")
    
def load_config():
    '''
    Loading up my super detailed config gile
    '''
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
model_id = config['model']['name']
dtype = getattr(torch, config['model'].get('dtype', 'float32'))

hf_token = os.environ.get('HF_TOKEN')

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,  
    token=hf_token 
)
    
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)  

section_names = {
    "issue_statements": "issue statement",
    "rule_statements": "rule statement",
    "case_comparisons": "case comparison",
    "rule_applications": "rule application",
    "writing_eval": "writing evaluation"
}


output_format = {
    "issue_statements": 
    """
    Your evaluation must follow this exact structure:
    Issue Statement Comments: This issue statement is [proficient/developing/beginning]. 
    [1-2 paragraphs of detailed feedback that includes specific strengths and areas for improvement. 
    Be sure to address format, content, length, structure, clarity, legal accuracy, and factual presentation as appropriate. 
    Provide specific examples from the text to support your assessment.]
    """,

    "rule_statements": 
    """
    Your evaluation must follow this exact structure:
    Rule Statement Comments: This rule statement is [proficient/developing/beginning]. 
    [1-2 paragraphs of detailed feedback that includes specific strengths and areas for improvement. 
    Be sure to address structure, authority, clarity, integration, and balance as appropriate. 
    Provide specific examples from the text to support your assessment.]
    """,

    "case_comparisons": 
    """
    Your evaluation must follow this exact structure:
    Explanation Comments: This explanation section is [proficient/developing/beginning]. 
    [1-2 paragraphs of detailed feedback that includes specific strengths and areas for improvement. 
    Be sure to address case selection, structure, depth, content, and balance as appropriate. 
    Provide specific examples from the text to support your assessment.]
    """,

    "rule_applications": 
    """
    Your evaluation must follow this exact structure:
    Application Comments: This application section is [proficient/developing/beginning]. 
    [1-2 paragraphs of detailed feedback that includes specific strengths and areas for improvement. 
    Be sure to address analogical reasoning, fact comparison, structure, completeness, and counterarguments as appropriate. 
    Provide specific examples from the text to support your assessment.]
    """,

    # There was a validation compinent for which some data was generate but there was
    # already too much going on and I was not confident in my ability to evaluate
    # memo writing on the whole
    # "writing_eval": 
    # """
    # Your evaluation must follow this exact structure:
    # Comments Memo Structure & Writing Mechanics: The memo structure and writing mechanics component is [proficient/developing/beginning]. 
    # [1-2 paragraphs of detailed feedback that includes specific strengths and areas for improvement. 
    # Be sure to address organization, paragraph structure, topic sentences, logical flow, and clarity as appropriate.
    # Provide specific examples from the text to support your assessment.]
    # """
}

def load_examples(section: str, shots: int) -> list:
    '''
    gets n random annotated examples from the eval folder (max n=5)
    each ecample is structure like:
    <student text>
    ###### (nifty delimiter)
    <rate sheet>
    <written feedback>
    '''
    data_path = Path(__file__).parent.parent / config['data']['evaluation'][section]
    files = list(data_path.glob("*.txt"))
        
    data = random.sample(files, min(shots, len(files)))
    examples = []
    for file in data:
        try:
            with open(file, "r") as f:
                content = f.read()
                parts = content.split("######") 
                if len(parts) != 2:
                    print(f"Invalid format in {file}")
                    continue
                text = parts[0].strip()
                feedback = parts[1].strip()
                examples.append({"text": text, "feedback": feedback})
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    return examples

def evaluate(section: str, text: str, shots: int) -> str:
    '''
    few shot eval on an ireac section using n examples retrived from load examples
    for in-context learning. 
    constructs a prompt w/ instruction, examples, and the text to evaluate
    returns model feedback
    '''
    try:
        prompt_path = Path(__file__).parent.parent / config['prompts'][section]
        with open(prompt_path, "r") as file:
            evaluation_guide = file.read()
            
        if section in output_format:
            format_template = output_format[section]
            evaluation_guide = evaluation_guide + "\n\n" + format_template
            
        examples = load_examples(section, shots)
        message = [{"role": "system", "content": evaluation_guide}]
        for i, example in enumerate(examples, 1):
            ex = f"EXAMPLE {i}:\n{example['text']}"
            message.append({"role": "user", "content": ex})
            message.append({"role": "assistant", "content": example['feedback']})
        
        section_name = section_names[section]
        message.append({"role": "user", "content": f"Now, please evaluate the following {section_name}:\n\n{text}\n\n===== PROVIDE YOUR EVALUATION BELOW THIS LINE ====="})
        
        inputs = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        input_device = next(model.parameters()).device
        inputs = {k: v.to(input_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            **config['generation']
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the part after the delimiter if it exists in the response
        delimiter = "===== PROVIDE YOUR EVALUATION BELOW THIS LINE ====="
        if delimiter in response:
            response_parts = response.split(delimiter)
            if len(response_parts) > 1:
                response = response_parts[1].strip()
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

'''
Evaluate Everything!
'''
def evaluate_issue_statement(text: str, num_shots: int) -> str:
    return evaluate("issue_statements", text, num_shots)

def evaluate_rule_statement(text: str, num_shots: int) -> str:
    return evaluate("rule_statements", text, num_shots)

def evaluate_case_comparison(text: str, num_shots: int) -> str:
    return evaluate("case_comparisons", text, num_shots)

def evaluate_rule_application(text: str, num_shots: int) -> str:
    return evaluate("rule_applications", text, num_shots)

def evaluate_writing_eval(text: str, num_shots: int) -> str:
    return evaluate("writing_eval", text, num_shots)