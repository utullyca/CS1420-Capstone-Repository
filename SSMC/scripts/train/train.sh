#!/bin/bash
#SBATCH --job-name=softsrv_train
#SBATCH --output=/home/%u/capstone/SoftSRV/logs/out/train-%j.out
#SBATCH --error=/home/%u/capstone/SoftSRV/logs/err/train-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=gpus
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --mail-type=BEGIN,END,FAIL

set -e
set -x

# for ddp
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(( 10000 + $RANDOM % 20000 ))
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=4
export TMPDIR=$HOME/tmp/$SLURM_JOB_ID
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p $TMPDIR

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source secrets.sh

CAPSTONE_DIR="$HOME/capstone"
SOFTSRV_DIR="$CAPSTONE_DIR/SoftSRV"
SCRIPT_PATH="$SOFTSRV_DIR/scripts/train.py"

echo "=== JOB INFO ==="
echo " Job started at: $(date)"
echo " Running on: $(hostname)"
echo " GPUs available: $(nvidia-smi --list-gpus | wc -l)"

source "$HOME/softsrv_env/bin/activate"
python --version
pip --version

echo "Starting training with 4 GPUs..."

SECTION=${1:-"rule_statements"} # can be changed to various configs 
ENCODER=${2:-"bert"}

echo " Training section: $SECTION"
echo " Encoder: $ENCODER"

torchrun \
    --nnodes=1 \
    --nproc_per_node=$SLURM_NTASKS \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:13300 \
    $SCRIPT_PATH \
    --section $SECTION \
    --encoder $ENCODER

echo "Training complete!"

if [ -d "$TMPDIR" ]; then
    echo "Cleaning up temporary directory: $TMPDIR"
    rm -rf $TMPDIR
fi