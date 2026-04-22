#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 0-02:00:00
# Output files
#SBATCH --error=./error/test_exp_%J.err
#SBATCH --output=./output/test_exp_%J.out

# Load modules
module purge
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
module load scikit-image/0.19.3-foss-2022a
module load scikit-learn/1.1.2-foss-2022a

# Activate venv
cd /mimer/NOBACKUP/groups/naiss2023-6-336/iele_2/project_2/venv_prova
source bin/activate

# Executes the code 
cd /mimer/NOBACKUP/groups/naiss2023-6-336/iele_2/project_2

# cache
# --cache-root timeSeries/cache_filtered/val_chopped_avg_NDVI_all \
# --cache-root timeSeries/cache_interpolated/val_chopped_avg_NDVI_clear_sky 



# IID CACHE NO INTERPOLAZIONE alfa=0.1
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/iid_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/NDVI_clear_sky_quantile_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/other_cache_no_interp/iid_chopped_avg_NDVI_clear_sky_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json


# IID CACHE NO INTERPOLAZIONE alfa=0.5
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/iid_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/NDVI_clear_sky_quantile/checkpoint_best.pth \
  --output-dir checkpoints/other_cache_no_interp/iid_chopped_avg_NDVI_clear_sky \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json

# history
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_history/iid_chopped_avg_NDVI_clear_sky_1_hist \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_hist_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_hist_alfa_0.1/iid_chopped_avg_1_hist_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_history/train_avg_NDVI_clear_sky_1_hist/scaler.json

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_history/iid_chopped_avg_NDVI_clear_sky_2_hist \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_hist_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_hist_alfa_0.1/iid_chopped_avg_2_hist_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_history/train_avg_NDVI_clear_sky_2_hist/scaler.json

# future
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_forecast/iid_chopped_avg_NDVI_clear_sky_1_forecast \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_forecast_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_forecast_alfa_0.1/iid_chopped_avg_1_forecast_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_forecast/train_avg_NDVI_clear_sky_1_forecast/scaler.json

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_forecast/iid_chopped_avg_NDVI_clear_sky_2_forecast \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_forecast_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_forecast_alfa_0.1/iid_chopped_avg_2_forecast_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_forecast/train_avg_NDVI_clear_sky_2_forecast/scaler.json


# OOD-S CACHE alfa=0.1
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/ood-s_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/NDVI_clear_sky_quantile_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/other_cache_no_interp/ood-s_chopped_avg_NDVI_clear_sky_0.1 \
  --batch-size "64" \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json

# OOD-S CACHE, alfa=0.5

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/ood-s_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/NDVI_clear_sky_quantile/checkpoint_best.pth \
  --output-dir checkpoints/other_cache_no_interp/ood-s_chopped_avg_NDVI_clear_sky \
  --batch-size "64" \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json

# history

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_history/ood-s_chopped_avg_NDVI_clear_sky_1_hist \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_hist_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_hist_alfa_0.1/ood-s_chopped_avg_1_hist_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_history/train_avg_NDVI_clear_sky_1_hist/scaler.json

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_history/ood-s_chopped_avg_NDVI_clear_sky_2_hist \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_hist_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_hist_alfa_0.1/ood-s_chopped_avg_2_hist_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_history/train_avg_NDVI_clear_sky_2_hist/scaler.json

# future
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_forecast/ood-s_chopped_avg_NDVI_clear_sky_1_forecast \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_forecast_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_forecast_alfa_0.1/ood-s_chopped_avg_1_forecast_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_forecast/train_avg_NDVI_clear_sky_1_forecast/scaler.json

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_forecast/ood-s_chopped_avg_NDVI_clear_sky_2_forecast \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_forecast_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_forecast_alfa_0.1/ood-s_chopped_avg_2_forecast_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_forecast/train_avg_NDVI_clear_sky_2_forecast/scaler.json


# OOD-T CACHE alfa = 0.1
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/ood-t_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/NDVI_clear_sky_quantile_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/other_cache_no_interp/ood-t_chopped_avg_NDVI_clear_sky_alfa_0.1 \
  --batch-size "64" \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json

# OOD-T CACHE alfa = 0.5
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/ood-t_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/NDVI_clear_sky_quantile/checkpoint_best.pth \
  --output-dir checkpoints/other_cache_no_interp/ood-t_chopped_avg_NDVI_clear_sky \
  --batch-size "64" \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json

# history

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_history/ood-t_chopped_avg_NDVI_clear_sky_1_hist \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_hist_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_hist_alfa_0.1/ood-t_chopped_avg_1_hist_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_history/train_avg_NDVI_clear_sky_1_hist/scaler.json

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_history/ood-t_chopped_avg_NDVI_clear_sky_2_hist \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_hist_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_hist_alfa_0.1/ood-t_chopped_avg_2_hist_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_history/train_avg_NDVI_clear_sky_2_hist/scaler.json

# future
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_forecast/ood-t_chopped_avg_NDVI_clear_sky_1_forecast \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_forecast_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_forecast_alfa_0.1/ood-t_chopped_avg_1_forecast_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_forecast/train_avg_NDVI_clear_sky_1_forecast/scaler.json

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_forecast/ood-t_chopped_avg_NDVI_clear_sky_2_forecast \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_forecast_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_forecast_alfa_0.1/ood-t_chopped_avg_2_forecast_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_forecast/train_avg_NDVI_clear_sky_2_forecast/scaler.json


# OD-ST CACHE alfa=0.1
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/ood-st_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/NDVI_clear_sky_quantile_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/other_cache_no_interp/ood-st_chopped_avg_NDVI_clear_sky_alfa_0.1 \
  --batch-size "64" \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json

# OD-ST CACHE alfa=0.5
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/ood-st_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/NDVI_clear_sky_quantile/checkpoint_best.pth \
  --output-dir checkpoints/other_cache_no_interp/ood-st_chopped_avg_NDVI_clear_sky \
  --batch-size "64" \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json

# history

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_history/ood-st_chopped_avg_NDVI_clear_sky_1_hist \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_hist_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_hist_alfa_0.1/ood-st_chopped_avg_1_hist_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_history/train_avg_NDVI_clear_sky_1_hist/scaler.json

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_history/ood-st_chopped_avg_NDVI_clear_sky_2_hist \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_hist_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_hist_alfa_0.1/ood-st_chopped_avg_2_hist_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_history/train_avg_NDVI_clear_sky_2_hist/scaler.json

# future
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_forecast/ood-st_chopped_avg_NDVI_clear_sky_1_forecast \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_forecast_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_forecast_alfa_0.1/ood-st_chopped_avg_1_forecast_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_forecast/train_avg_NDVI_clear_sky_1_forecast/scaler.json

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_forecast/ood-st_chopped_avg_NDVI_clear_sky_2_forecast \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_forecast_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_forecast_alfa_0.1/ood-st_chopped_avg_2_forecast_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_forecast/train_avg_NDVI_clear_sky_2_forecast/scaler.json

# VAL_CHOPPED CACHE
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/val_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/NDVI_clear_sky_quantile/checkpoint_best.pth \
  --output-dir checkpoints/other_cache_no_interp/val_chopped_avg_NDVI_clear_sky \
  --batch-size "64" \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json

# history
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_history/val_chopped_avg_NDVI_clear_sky_1_hist \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_hist_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_hist_alfa_0.1/val_chopped_avg_1_hist__alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_history/train_avg_NDVI_clear_sky_1_hist/scaler.json

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_history/val_chopped_avg_NDVI_clear_sky_2_hist \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_hist_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_hist_alfa_0.1/val_chopped_avg_2_hist_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_history/train_avg_NDVI_clear_sky_2_hist/scaler.json

# future
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_forecast/val_chopped_avg_NDVI_clear_sky_1_forecast \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_forecast_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_1_forecast_alfa_0.1/val_chopped_avg_1_forecast_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_forecast/train_avg_NDVI_clear_sky_1_forecast/scaler.json

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d_ablation_forecast/val_chopped_avg_NDVI_clear_sky_2_forecast \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_forecast_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp_ablation_window/NDVI_clear_sky_quantile_2_forecast_alfa_0.1/val_chopped_avg_2_forecast_alfa_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d_ablation_forecast/train_avg_NDVI_clear_sky_2_forecast/scaler.json

# VAL_CHOPPED CACHE GNDVI
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/val_chopped_avg_GNDVI_clear_sky \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/GNDVI_clear_sky_quantile/checkpoint_best.pth \
  --output-dir checkpoints/other_cache_no_interp/val_chopped_avg_GNDVI_clear_sky \
  --batch-size "64" \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_GNDVI_clear_sky/scaler.json

# VAL_CHOPPED CACHE EVI
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/val_chopped_avg_EVI_clear_sky \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/EVI_clear_sky_quantile/checkpoint_best.pth \
  --output-dir checkpoints/other_cache_no_interp/val_chopped_avg_EVI_clear_sky \
  --batch-size "64" \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_EVI_clear_sky/scaler.json

# VAL_CHOPPED NO ft NO ALFA
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/val_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/ablation_15d_small_no_interp/NDVI_clear_sky_quantile_no_ft_no_alfa/checkpoint_best.pth \
  --output-dir checkpoints/ablation_15d_small_no_interp/val_chopped_avg_NDVI_clear_sky_no_ft_no_alfa \
  --batch-size "64" \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering false \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json

# VAL_CHOPPED NO FT si ALFA
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/val_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/ablation_15d_small_no_interp/NDVI_clear_sky_quantile_no_ft_si_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/ablation_15d_small_no_interp/val_chopped_avg_NDVI_clear_sky_no_ft_si_alfa_0.1 \
  --batch-size "64" \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering false \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json




# VAL_CHOPPED SI FT si ALFA
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/val_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/ablation_15d_small_no_interp/NDVI_clear_sky_quantile_si_ft_no_alfa/checkpoint_best.pth \
  --output-dir checkpoints/ablation_15d_small_no_interp/val_chopped_avg_NDVI_clear_sky_si_ft_no_alfa \
  --batch-size "64" \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json

# VAL_CHOPPED con alfa=0.1
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/val_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/NDVI_clear_sky_quantile_alfa_0.1/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/val_chopped_avg_NDVI_clear_sky_alfa_0.1 \
  --batch-size "64" \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json

# VAL_CHOPPED con alfa=0.2
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/val_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/NDVI_clear_sky_quantile_alfa_0.2/checkpoint_best.pth \
  --output-dir checkpoints/sinh_crop_60_weighted_loss_ft_eng_no_interp/val_chopped_avg_NDVI_clear_sky_alfa_0.2 \
  --batch-size "64" \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --feature-engineering true \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json


########### ABLATION 15 DAYS

# ablate history
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/val_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/ablation_15d_small_no_interp/NDVI_clear_sky_quantile_ablate-history-covariate_0.1/checkpoint_best.pth \
  --output-dir checkpoints/ablation_15d_small_no_interp/val_chopped_avg_NDVI_clear_sky_ablate-history-covariate_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --ablate-future-covariates false \
  --ablate-history-covariates true \
  --ablate-target-history false \
  --feature-engineering true \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json

# ablate future
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/val_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/ablation_15d_small_no_interp/NDVI_clear_sky_quantile_ablate-future-covariates_0.1/checkpoint_best.pth \
  --output-dir checkpoints/ablation_15d_small_no_interp/val_chopped_avg_NDVI_clear_sky_ablate-future-covariates_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --ablate-future-covariates true \
  --ablate-history-covariates false \
  --ablate-target-history false \
  --feature-engineering true \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json


# ablate target
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/val_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/ablation_15d_small_no_interp/NDVI_clear_sky_quantile_ablate-target-history_0.1/checkpoint_best.pth \
  --output-dir checkpoints/ablation_15d_small_no_interp/val_chopped_avg_NDVI_clear_sky_ablate-target-history_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --ablate-future-covariates false \
  --ablate-history-covariates false \
  --ablate-target-history true \
  --feature-engineering true \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json


# ablate-future-target
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/val_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/ablation_15d_small_no_interp/NDVI_clear_sky_quantile_ablate-future-target_0.1/checkpoint_best.pth \
  --output-dir checkpoints/ablation_15d_small_no_interp/val_chopped_avg_NDVI_clear_sky_quantile_ablate-future-target_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --ablate-future-covariates true \
  --ablate-history-covariates false \
  --ablate-target-history  true \
  --feature-engineering true \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json



# ablate-future-history
python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/val_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/ablation_15d_small_no_interp/NDVI_clear_sky_quantile_ablate-future-history_0.1/checkpoint_best.pth \
  --output-dir checkpoints/ablation_15d_small_no_interp/val_chopped_avg_NDVI_clear_sky_quantile_ablate-future-history_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --ablate-future-covariates true \
  --ablate-history-covariates true \
  --ablate-target-history false \
  --feature-engineering true \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json

# ablate-history-target_0.1

python agrimatnet/test_quantile_ablation.py \
  --cache-root timeSeries/cache_monthly_noise_15d/val_chopped_avg_NDVI_clear_sky \
  --weights checkpoints/ablation_15d_small_no_interp/NDVI_clear_sky_quantile_ablate-history-target_0.1/checkpoint_best.pth \
  --output-dir checkpoints/ablation_15d_small_no_interp/val_chopped_avg_NDVI_clear_sky_quantile_ablate-history-target_0.1 \
  --batch-size "64" \
  --d-model "128" \
  --num-layers "8" \
  --num-heads "8" \
  --dim-feedforward "512" \
  --dropout "0.1" \
  --quantiles 0.1,0.5,0.9 \
  --ablate-future-covariates false \
  --ablate-history-covariates true \
  --ablate-target-history true \
  --feature-engineering true \
  --scaler-path timeSeries/cache_monthly_noise_15d/train_avg_NDVI_clear_sky/scaler.json
