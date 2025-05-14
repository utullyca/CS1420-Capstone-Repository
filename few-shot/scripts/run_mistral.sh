#!/bin/bash
#SBATCH --job-name=mistral_fewshot
#SBATCH --output=/home/%u/capstone/FewShot/logs/out/mistral-fewshot-%j.out
#SBATCH --error=/home/%u/capstone/FewShot/logs/err/mistral-fewshot-%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpus
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mail-type=BEGIN,END,FAIL

set -e
set -x

export TMPDIR=$HOME/tmp/$SLURM_JOB_ID # More memory management
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p $TMPDIR

export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

source secrets.sh # API key goes here
CAPSTONE_DIR="$HOME/capstone"
FEWSHOT_DIR="$CAPSTONE_DIR/FewShot"
SCRIPT_PATH="$FEWSHOT_DIR/scripts/run_mistral.py"
mkdir -p "$FEWSHOT_DIR/logs/out"
mkdir -p "$FEWSHOT_DIR/logs/err"

echo "=== JOB INFO ==="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "User: $(whoami)"

source "$HOME/mistral_env/bin/activate"

echo "=== RUNNING MISTRAL FEW-SHOT EVALUATION ==="
cd "$FEWSHOT_DIR"
python "$SCRIPT_PATH"

rm -rf $TMPDIR
