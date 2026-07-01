<div align="center">
<h1>Probabilistic NDVI Forecasting from Sparse Satellite Time Series and Weather Covariates
</h1>

[Irene Iele](https://scholar.google.com/citations?user=srLH7lkAAAAJ&hl=it&oi=ao)<sup>1</sup>, 
[Giulia Romoli](https://scholar.google.com/citations?user=mSFVXpIAAAAJ&hl=it&oi=ao)<sup>2</sup>, 
[Daniele Molino](https://scholar.google.com/citations?user=MxxVQxoAAAAJ&hl=it&oi=ao)<sup>1</sup>, 
[Elena Mulero Ayllón](https://scholar.google.com/citations?user=-BOMvaUAAAAJ&hl=it&oi=ao)<sup>1</sup>, 
[Filippo Ruffini](https://scholar.google.com/citations?user=eW7C8YMAAAAJ&hl=it&oi=ao)<sup>1,2</sup>, 
[Paolo Soda](https://scholar.google.com/citations?user=E7rcYCQAAAAJ&hl=it&oi=ao)<sup>1,2</sup>, 
[Matteo Tortora](https://matteotortora.github.io)<sup>3</sup>

<sup>1</sup>  University Campus Bio-Medico of Rome,
<sup>2</sup>  Umeå University,
<sup>3</sup>  University of Genoa
</div>

# Overview
Public release of the code for NDVI forecasting using satellite time series and weather data.

<div align="center">
  <img src="method_avss.png" alt="Method overview" width="900"/>
</div>

<div align="center">
<a href="https://arxiv.org/abs/2602.17683">
  <img src="https://img.shields.io/badge/arXiv-2602.17683-B31B1B?style=flat&logo=arxiv&logoColor=white"/>
</a>
</div>

<br/>

## Contact
For questions and comments, feel free to contact me: irene.iele@unicampus.it

## Citation
If you use this [work](https://arxiv.org/abs/2602.17683), please cite:

```bibtex
@article{iele2026probabilistic,
  title={Probabilistic NDVI Forecasting from Sparse Satellite Time Series and Weather Covariates},
  author={Iele, Irene and Romoli, Giulia and Molino, Daniele and Ayll{\'o}n, Elena Mulero and Ruffini, Filippo and Soda, Paolo and Tortora, Matteo},
  journal={arXiv preprint arXiv:2602.17683},
  year={2026}
}
```

## Main Scripts

- `dataset_builder/cache_builder_unified_monthly_noise.py`: builds the cached dataset from the raw CSV files, adding monthly weather noise and saving the cache metadata and scaler.
- `dataset_builder/torch_dataset.py`: loads the cached samples as a PyTorch dataset, with optional feature engineering, scaling, and target discretization.
- `dataset_builder/scaler.py`: handles feature and target scaling, plus inverse transforms for evaluation.
- `agrimatnet/model_quantile.py`: defines the AgriMatNet quantile model and the pinball loss used during training.
- `agrimatnet/train_quantile_ablation.py`: trains the quantile model and supports ablation studies and time-weighted loss.
- `agrimatnet/test_quantile_ablation.py`: evaluates a trained checkpoint and exports metrics, predictions, and plots.
- `agrimatnet/train_quantile.bash`: SLURM wrapper that launches the training job on the cluster.
- `agrimatnet/test_quantile_small_no_interp.bash`: SLURM wrapper that launches the evaluation job on the cluster for the no-interpolation experiments.
