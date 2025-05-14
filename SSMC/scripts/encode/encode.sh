#!/bin/bash
#SBATCH --job-name=encode
#SBATCH --output=encode-%j.out
#SBATCH --error=encode-%j.err
#SBATCH --partition=gpus
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL

set -e
set -x
CAPSTONE_DIR="$HOME/capstone"
SOFTSRV_DIR="$CAPSTONE_DIR/SoftSRV"
SCRIPT_PATH="$SOFTSRV_DIR/scripts/encode.py"

echo "=== JOB INFO ==="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "User: $(whoami)"

mkdir -p "$SOFTSRV_DIR/data/embeddings"

python3 -m venv "$HOME/mistral_env"
source "$HOME/mistral_env/bin/activate"
python --version
pip --version

pip install transformers pyyaml

echo "=== RUNNING ENCODING ==="
cd "$CAPSTONE_DIR"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script not found at $SCRIPT_PATH"
    ls -la "$SOFTSRV_DIR/scripts/"
    exit 1
fi

python "$SCRIPT_PATH"

