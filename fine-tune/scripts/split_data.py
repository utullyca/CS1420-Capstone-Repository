import random
import shutil
from pathlib import Path

base_dir = Path("/Users/uly/Desktop/Capstone/FineTune/data")
section_mappings = {
    "1_issue_statements": "1_issue_statements",
    "2_rule_statements": "2_rule_statements",
    "3_case_comparisons": "3_case_comparisons",
    "4_rule_applications": "4_rule_applications",
    "5_writing": "5_writing"
}

for section, _ in section_mappings.items():
    eval_path = base_dir / "evals" / section
    valid_path = base_dir / "validation" / section
    valid_path.mkdir(exist_ok=True, parents=True)   
    files = list(eval_path.glob("*.txt"))
    files_to_move = random.sample(files, 3)
    for file in files_to_move:
        dest_file = valid_path / file.name
        print(f"Moving {file.name} from evals to validation")
        shutil.move(file, dest_file)
print("\nSplit complete!")
