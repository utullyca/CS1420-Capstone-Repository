#!/bin/bash
#SBATCH --job-name=finetune_train
#SBATCH --output=/home/%u/capstone/FineTune/logs/out/train-%j.out
#SBATCH --error=/home/%u/capstone/FineTune/logs/err/train-%j.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

# added to debug cuda errors
# export NCCL_DEBUG=INFO
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

source secrets.sh
CAPSTONE_DIR="$HOME/capstone"
FINETUNE_DIR="$CAPSTONE_DIR/FineTune"
SCRIPT_PATH="$FINETUNE_DIR/scripts/train.py"

mkdir -p $CAPSTONE_DIR/FineTune/logs/out
mkdir -p $CAPSTONE_DIR/FineTune/logs/err

echo "=== JOB INFO ==="
echo " Job started at: $(date)"
echo " Running on: $(hostname)"
echo " GPUs available: $(nvidia-smi --list-gpus | wc -l)"


source "$HOME/softsrv_env/bin/activate"
python --version
pip --version

pip install --no-cache-dir transformers datasets peft accelerate bitsandbytes sentencepiece tqdm --upgrade --no-deps


echo "Script path: $SCRIPT_PATH"
ls -la $SCRIPT_PATH

echo "Starting LoRA fine-tuning for all sections..."

export PYTHONUNBUFFERED=1
python -u $SCRIPT_PATH

echo "Training complete!"
if [ -d "$TMPDIR" ]; then
  rm -rf $TMPDIR
fi