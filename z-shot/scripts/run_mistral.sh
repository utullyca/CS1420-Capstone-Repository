#!/bin/bash
#SBATCH --job-name=mistral
#SBATCH --output=mistral-%j.out
#SBATCH --error=mistral-%j.err
#SBATCH --partition=gpus
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL

set -e
set -x

CAPSTONE_DIR="$HOME/capstone"
ZEROSHOT_DIR="$CAPSTONE_DIR/ZeroShot"
SCRIPT_PATH="$ZEROSHOT_DIR/scripts/run_mistral.py"

echo "=== JOB INFO ==="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "User: $(whoami)"

mkdir -p "$ZEROSHOT_DIR/outputs/issue" "$ZEROSHOT_DIR/outputs/rule" 
mkdir -p "$ZEROSHOT_DIR/outputs/case_comp" "$ZEROSHOT_DIR/outputs/rule_app"


python3 -m venv "$HOME/mistral_env"
source "$HOME/mistral_env/bin/activate"


pip install -r "$ZEROSHOT_DIR/requirements.txt"

source secret.sh # api key used to go here

echo "=== RUNNING MISTRAL ==="
cd "$CAPSTONE_DIR"
EVAL_TYPE=${1:-"all"}

python "$SCRIPT_PATH"

echo "Job completed at: $(date)"