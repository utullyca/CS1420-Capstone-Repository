#!/bin/bash
#SBATCH --job-name=finetune_eval
#SBATCH --output=/home/%u/capstone/FineTune/logs/out/eval-%j.out
#SBATCH --error=/home/%u/capstone/FineTune/logs/err/eval-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpus
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mail-type=BEGIN,END,FAIL

set -e
set -x

export TMPDIR=$HOME/tmp/$SLURM_JOB_ID
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p $TMPDIR

# export NCCL_DEBUG=INFO
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

source secrets.sh
CAPSTONE_DIR="$HOME/capstone"
FINETUNE_DIR="$CAPSTONE_DIR/FineTune"
SCRIPT_PATH="$FINETUNE_DIR/scripts/evaluate.py"

mkdir -p $CAPSTONE_DIR/FineTune/logs/out
mkdir -p $CAPSTONE_DIR/FineTune/logs/err

echo "=== JOB INFO ==="
echo " Job started at: $(date)"
echo " Running on: $(hostname)"
echo " GPUs available: $(nvidia-smi --list-gpus | wc -l)"


source "$HOME/softsrv_env/bin/activate"
python --version
pip --version

echo "Starting evaluation of fine-tuned models..."

python $SCRIPT_PATH

echo "Evaluation complete!"
if [ -d "$TMPDIR" ]; then
  echo "Cleaning up temporary directory: $TMPDIR"
  rm -rf $TMPDIR
fi