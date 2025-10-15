#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true
#$ -R y
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N vgg13_masked
#$ -t 1-6
set -euo pipefail

hostname
date

number=$SGE_TASK_ID
paramfile="scripts/jobs_vgg13.txt"

# ---------------------------------------------------------------------
# 1.  Load toolchains and activate virtual-env
# ---------------------------------------------------------------------
source /share/apps/source_files/python/python-3.9.5.source
source /share/apps/source_files/cuda/cuda-11.8.source
source /SAN/intelsys/syn_vae_datasets/MATS_anti_spur/spur_venv/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# ---------------------------------------------------------------------
# 2.  Keep Matplotlib out of your home quota
# ---------------------------------------------------------------------
export MPLCONFIGDIR="$TMPDIR/mplcache"
mkdir -p "$MPLCONFIGDIR"

# ---------------------------------------------------------------------
# 3.  Create output directories
# ---------------------------------------------------------------------
mkdir -p checkpoints
mkdir -p results
mkdir -p data
mkdir -p plots

# ---------------------------------------------------------------------
# 4.  Extract task-specific parameters
# ---------------------------------------------------------------------
# Format: mode seed
mode=$(sed -n ${number}p "$paramfile" | awk '{print $1}')
seed=$(sed -n ${number}p "$paramfile" | awk '{print $2}')

date
echo "Training: VGG13 | Mode: $mode | Seed: $seed"

# ---------------------------------------------------------------------
# 5.  Run training
# ---------------------------------------------------------------------
echo "Starting training..."
python3.9 -u train.py \
    --arch vgg13 \
    --mode $mode \
    --seed $seed \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.1 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --n_masks 3 \
    --lambda_max 1.0 \
    --num_workers 4 \
    --device cuda \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints \
    --results_dir ./results

date
echo "Training completed successfully: VGG13 | Mode: $mode | Seed: $seed"
