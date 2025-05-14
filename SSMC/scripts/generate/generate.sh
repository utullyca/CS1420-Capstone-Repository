#!/bin/bash
#SBATCH --job-name=softsrv_generate
#SBATCH --output=/home/%u/capstone/SoftSRV/logs/out/generate-%j.out
#SBATCH --error=/home/%u/capstone/SoftSRV/logs/err/generate-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/capstone/secrets.sh
CAPSTONE_DIR="$HOME/capstone"
SOFTSRV_DIR="$CAPSTONE_DIR/SoftSRV"
SCRIPT_PATH="$SOFTSRV_DIR/scripts/generate/generate.py"

echo "=== JOB INFO ==="
echo " Job started at: $(date)"
echo " Running on: $(hostname)"
echo " GPUs available: $(nvidia-smi --list-gpus | wc -l)"

source "$HOME/softsrv_env/bin/activate"
python --version
pip --version

python $SCRIPT_PATH

echo "Generation complete!"
if [ -d "$TMPDIR" ]; then
  echo "Cleaning up temporary directory: $TMPDIR"
  rm -rf $TMPDIR
fi