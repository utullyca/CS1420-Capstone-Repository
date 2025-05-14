from pathlib import Path
from evaluate import evaluate_rule_application, evaluate_rule_statement, evaluate_issue_statement, evaluate_case_comparison
import sys
import time
import yaml

sys.path.append(str(Path(__file__).parent.parent.parent))

def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Debugging print lines for evaluating section and file (I had lots of data path issues)
def read_files(input, eval, output):
    outputs = Path(__file__).parent.parent / "outputs" / output
    outputs.mkdir(parents=True, exist_ok=True)
    files = list(input.glob("*.txt"))

    for f in files:
        print(f"Reading {f}")
        with open(f, "r") as file:
            text = file.read()

        try: 
            # Evaluate and save output
            print(f"Evaluating {f}")
            result = eval(text)
            daydate = time.strftime("%Y-%m-%d")
            output = outputs / (f.name + "_" + daydate + ".txt")
            with open(output, "w") as file:
                file.write(result)
            print(f"Saved to {output}")
        except Exception as e:
            print(f"Error evaluating {f}: {str(e)}")


def run_z_shot(config):
    '''
        Runs z-shot on all the IREAC section
    '''
    evals = {
        "issue": {"eval": evaluate_issue_statement, "input": Path(config["data"]["issue_statements"]), "output": "issue"},
        "rule": {"eval": evaluate_rule_statement, "input": Path(config["data"]["rule_statements"]), "output": "rule"},
        "case_comp": {"eval": evaluate_case_comparison, "input": Path(config["data"]["case_comparisons"]), "output": "case_comp"},
        "rule_app": {"eval": evaluate_rule_application, "input": Path(config["data"]["rule_applications"]), "output": "rule_app"}
    }

    for eval, data in evals.items():
        print(f"Running {eval} evaluation")
        read_files(data["input"], data["eval"], data["output"])


# def run_one_eval(config, section):
#     '''
#         Runs z-shot on one specifc the IREAC section
#     ARGS:
#         The section to run z-shot on
#     '''
#     evals = {
#         "issue": {"eval": evaluate_issue_statement, "input": Path(config["data"]["issue_statements"]), "output": "issue"},
#         "rule": {"eval": evaluate_rule_statement, "input": Path(config["data"]["rule_statements"]), "output": "rule"},
#         "case_comp": {"eval": evaluate_case_comparison, "input": Path(config["data"]["case_comparisons"]), "output": "case_comp"},
#         "rule_app": {"eval": evaluate_rule_application, "input": Path(config["data"]["rule_applications"]), "output": "rule_app"}
#     }

#     evaluate = evals[section]
#     print(f"Running {section} evaluation")
#     read_files(evaluate["input"], evaluate["eval"], evaluate["output"])

# Default is all sections bbut specific section runs were used for early prompt tweaking

if __name__ == "__main__":
    config = load_config()
    # run_one_eval(config, "issue")
    # run_one_eval(config, "rule")
    # run_one_eval(config, "case_comp")
    # run_one_eval(config, "rule_app")
    run_z_shot(config)


    


            