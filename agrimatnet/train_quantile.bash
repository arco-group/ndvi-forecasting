#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-662 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 0-10:00:00
# Output files
#SBATCH --error=./error/expName_%J.err
#SBATCH --output=./output/expName_s%J.out

# Load modules
module purge
module load CUDA/12.1.0
module load Python/3.10.8-GCCcore-12.2.0

# Activate venv
cd /mimer/NOBACKUP/groups/naiss2023-6-336/iele_2/project_2/venv_prova
source bin/activate

# Executes the code 
cd /mimer/NOBACKUP/groups/naiss2023-6-336/iele_2/project_2

# cache
#--cache-root timeSeries/cache_interpolated_monthly_noise_15d/train_avg_NDVI_clear_sky \
#--cache-root timeSeries/cache_interpolated_monthly_noise_10d/train_avg_NDVI_clear_sky \
#--cache-root timeSeries/cache_interpolated_monthly_noise_5d/train_avg_NDVI_clear_sky \



#name
# NDVI_clear_sky_quantile_interpolated
# NDVI_all_quantile_filtered

#NDVI_clear_sky_quantile_filtered_monthly_noise

# --d-model 128 \ #256
#--dim-feedforward 512 \ #1024


python agrimatnet/train_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky \
  --epochs 200 \
  --batch-size 128 \
  --lr 1e-4 \
  --train-split 0.8 \
  --seed 42 \
  --apply-scaling true \
  --min-crop-pixels 60 \
  --experiment-name sinh_crop_60_weighted_loss_ft_eng_no_interp/NDVI_clear_sky_quantile_alfa_0.1 \
  --quantiles 0.1,0.5,0.9 \
  --lr-reduce-on-plateau true \
  --lr-factor 0.2 \
  --lr-patience 20 \
  --lr-min 5e-5 \
  --d-model 128 \
  --num-layers 8 \
  --num-heads 8 \
  --dim-feedforward 512 \
  --dropout 0.1 \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --feature-engineering true \
  --time-weight-alpha 0.1 \

