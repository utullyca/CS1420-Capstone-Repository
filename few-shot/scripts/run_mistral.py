import random
from pathlib import Path
from evaluate import evaluate, load_config
import os
import gc
import torch
import time
from datetime import datetime

def clear_gpu_memory():
    '''
    Fancy mem cache clearing technique i implemented to prevent more email
    exchanges with the hydra cluster overlords
    '''
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)

def read_files(section):
    '''
    gets files in the validation section and run few shot (1,3,5) on each example
    '''
    outputs = Path(__file__).parent.parent / "outputs" / section
    outputs.mkdir(parents=True, exist_ok=True)
    
    config = load_config()
    data_path = Path(__file__).parent.parent / config['data']['validation'][section]
    files = list(data_path.glob("*.txt"))
    shot_counts = [1, 3, 5]

    for file in files:
        for shot_count in shot_counts:
            try:
                print(f"  Evaluating {file.stem} with {shot_count} shots...")
                with open(file, "r") as f:
                    text = f.read()
                
                result = evaluate(section, text, shot_count)
                
                output_file = outputs / f"{file.stem}_{section}_{shot_count}shot.txt"
                with open(output_file, "w") as f:
                    f.write(result)
                print(f"  Saved to {output_file}")                
                clear_gpu_memory()
                
            except Exception as e:
                print(f"  Error evaluating {file.stem} with {shot_count} shots: {str(e)}")
                clear_gpu_memory()
    
def main():
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sections = ["issue_statements", "rule_statements", "case_comparisons", "rule_applications", "writing_eval"]    
    print(f"\n=== Starting FewShot Evaluation ===") 
    for i, section in enumerate(sections, 1):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n=== [{current_time}] Evaluating section {i}/{5}: {section} ===")
        section_dir = output_dir / section
        section_dir.mkdir(parents=True, exist_ok=True)
        read_files(section)
        
        clear_gpu_memory()
    print(f"\n=== Completed FewShot Evaluation ===") 

if __name__ == "__main__":
    main()