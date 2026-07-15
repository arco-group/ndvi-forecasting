import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .cache_builder_unified import DatasetCacheBuilder


class MonthlyNoiseCacheBuilder(DatasetCacheBuilder):
    """
    Cache builder with weather noise based on monthly std (area-level, with global fallback)
    and linear growth with the real lead time (days).
    """

    def __init__(
        self,
        *args,
        min_samples_per_month=100,
        g5_target=2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_samples_per_month = int(min_samples_per_month)
        self.g5_target = float(g5_target)
        self.month_counts_area = defaultdict(lambda: defaultdict(int))
        self.month_counts_global = defaultdict(int)
        self.monthly_std_area = defaultdict(lambda: defaultdict(dict))
        self.monthly_std_global = defaultdict(dict)

    def build(self):
        self.compute_monthly_stats()
        super().build()

    def compute_monthly_stats(self):
        """
        Compute the monthly std for weather features.
        If an area does not have enough samples in a month, fall back to the global stats.
        """
        groups = self.collect_groups()
        if not groups:
            self.logger.warning("Nessun CSV trovato in %s", self.data_root)
            return

        sums_area = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        sumsq_area = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        count_area = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        sums_global = defaultdict(lambda: defaultdict(float))
        sumsq_global = defaultdict(lambda: defaultdict(float))
        count_global = defaultdict(lambda: defaultdict(int))

        for area_name in self.progress(sorted(groups.keys()), "Stats mensili"):
            for csv_path in groups[area_name]:
                dataframe = self.load_csv(csv_path)
                if dataframe is None or dataframe.empty:
                    continue
                feature_frame, _ = self.prepare_features(dataframe)
                if feature_frame is None or feature_frame.empty:
                    continue

                meteo_cols = [c for c in self.meteo_columns if c in feature_frame.columns]
                if not meteo_cols:
                    meteo_cols = list(feature_frame.columns)

                meteo_frame = feature_frame[meteo_cols]
                months = meteo_frame.index.month
                for month in range(1, 13):
                    month_mask = months == month
                    if not month_mask.any():
                        continue
                    month_slice = meteo_frame.loc[month_mask]
                    self.month_counts_area[area_name][month] += len(month_slice)
                    self.month_counts_global[month] += len(month_slice)
                    for col in meteo_cols:
                        values = month_slice[col].to_numpy(dtype=np.float64, copy=False)
                        values = values[~np.isnan(values)]
                        if values.size == 0:
                            continue
                        sums_area[area_name][month][col] += float(values.sum())
                        sumsq_area[area_name][month][col] += float((values ** 2).sum())
                        count_area[area_name][month][col] += int(values.size)
                        sums_global[month][col] += float(values.sum())
                        sumsq_global[month][col] += float((values ** 2).sum())
                        count_global[month][col] += int(values.size)

        def finalize_std(sums, sumsq, count):
            stds = {}
            for col, cnt in count.items():
                if cnt <= 1:
                    stds[col] = 1.0
                    continue
                mean = sums[col] / cnt
                var = (sumsq[col] / cnt) - (mean ** 2)
                stds[col] = float(math.sqrt(max(var, 0.0)))
            return stds

        for area_name, months in count_area.items():
            for month, col_counts in months.items():
                stds = finalize_std(
                    sums_area[area_name][month],
                    sumsq_area[area_name][month],
                    col_counts,
                )
                self.monthly_std_area[area_name][month] = stds

        for month, col_counts in count_global.items():
            stds = finalize_std(
                sums_global[month],
                sumsq_global[month],
                col_counts,
            )
            self.monthly_std_global[month] = stds

    def _get_monthly_std(self, area_name, month, feature_name):
        area_count = self.month_counts_area.get(area_name, {}).get(month, 0)
        if area_count >= self.min_samples_per_month:
            std = self.monthly_std_area.get(area_name, {}).get(month, {}).get(feature_name)
        else:
            std = None
        if std is None:
            std = self.monthly_std_global.get(month, {}).get(feature_name)
        if std is None or not np.isfinite(std) or std <= 0:
            std = 1.0
        return std

    def add_noise_to_future(self, future_slice, future_mask, future_timestamps, history_end_ts, area_name):
        noisy = future_slice.copy()
        noise_matrix = np.zeros_like(future_slice, dtype=np.float32)

        if self.noise_percentage is None or self.noise_percentage == 0:
            return noisy, noise_matrix

        num_features = future_slice.shape[1]
        columns_to_perturb = (
            [self.feature_names.index(name) for name in self.meteo_columns if name in self.feature_names]
            if self.feature_names
            else list(range(num_features))
        )
        if not columns_to_perturb:
            columns_to_perturb = list(range(num_features))

        valid_mask = ~future_mask[:, columns_to_perturb]
        if not valid_mask.any():
            return noisy, noise_matrix

        column_values = future_slice[:, columns_to_perturb]
        feature_names = [self.feature_names[idx] for idx in columns_to_perturb]

        future_ts = np.array(future_timestamps, dtype="datetime64[ns]")
        history_end = np.datetime64(history_end_ts)
        delta_days = (future_ts - history_end).astype("timedelta64[D]").astype(np.float32)
        delta_days = np.maximum(delta_days, 0.0)
        delta_t5 = float(delta_days.max()) if delta_days.size else 0.0
        if delta_t5 > 0:
            beta = (self.g5_target - 1.0) / delta_t5
        else:
            beta = 0.0
        growth = 1.0 + beta * delta_days
        growth = np.maximum(growth, 1.0)

        month_std_cache = {}
        for month in range(1, 13):
            month_std_cache[month] = np.array(
                [self._get_monthly_std(area_name, month, fname) for fname in feature_names],
                dtype=np.float32,
            )

        months = [pd.Timestamp(ts).month for ts in future_ts]
        std_matrix = np.vstack([month_std_cache[m] for m in months]).astype(np.float32)
        scale = std_matrix * self.noise_percentage * growth[:, None]

        noise = self.rng.normal(loc=0.0, scale=scale, size=column_values.shape).astype(np.float32)
        noise = np.where(valid_mask, noise, 0.0)
        noise_matrix[:, columns_to_perturb] = noise
        noisy[:, columns_to_perturb] = np.where(
            valid_mask,
            column_values + noise,
            column_values,
        )
        return noisy, noise_matrix

    def process_area(self, area_name, csv_files):
        """
        Build samples for a single area and save the npz cache plus metadata JSON.
        """
        samples_history = []
        samples_history_mask = []
        samples_history_timestamps = []
        samples_future = []
        samples_future_mask = []
        samples_future_noise = []
        samples_future_timestamps = []
        samples_target = []
        samples_target_mask = []
        samples_target_timestamps = []
        samples_future_target_positions = []
        samples_climate = []
        metadata_records = []

        climate_value = self.climate_map[area_name]

        for csv_path in self.progress(csv_files, f"{area_name}"):
            dataframe = self.load_csv(csv_path)
            if dataframe is None or dataframe.empty:
                continue

            feature_frame, target_series = self.prepare_features(dataframe)
            if feature_frame is None:
                continue

            columns = feature_frame.columns.tolist()
            history_target_name = f"{self.target_column}_history"
            if self.feature_names is None:
                self.base_feature_columns = columns
                self.feature_names = columns + [history_target_name]
                self.initialise_stats(len(self.feature_names))
            else:
                if columns != self.base_feature_columns:
                    self.logger.error("L'ordine delle colonne non coincide in %s", csv_path)
                    raise ValueError("Colonne incoerenti fra i CSV. Uniformare prima i dati.")

            feature_values = feature_frame.to_numpy(dtype=np.float32)
            feature_masks = np.isnan(feature_values)

            target_values = target_series.to_numpy(dtype=np.float32)

            timestamps = feature_frame.index.to_numpy()
            total_rows = len(feature_frame)

            valid_positions = [idx for idx in range(total_rows) if not math.isnan(target_values[idx])]
            if not valid_positions:
                continue
            if self.step <= 0:
                start_positions = valid_positions
            else:
                start_positions = [valid_positions[i] for i in range(0, len(valid_positions), self.step)]

            for start in start_positions:
                history_indices = []
                history_target_count = 0
                ptr = start

                while ptr < total_rows and history_target_count < self.history_window:
                    history_indices.append(ptr)
                    if not math.isnan(target_values[ptr]):
                        history_target_count += 1
                    ptr += 1

                if history_target_count < self.history_window:
                    continue

                history_slice = feature_values[history_indices]
                history_ts = timestamps[history_indices]

                target_history_column = np.full((len(history_indices), 1), np.nan, dtype=np.float32)
                for pos, original_idx in enumerate(history_indices):
                    value = target_values[original_idx]
                    if not math.isnan(value):
                        target_history_column[pos, 0] = value

                history_slice = np.concatenate([history_slice, target_history_column], axis=1)
                history_mask = np.isnan(history_slice)

                future_indices = []
                future_target_values = []
                future_target_timestamps = []
                future_target_positions = []
                while ptr < total_rows and len(future_target_values) < self.forecast_window:
                    future_indices.append(ptr)
                    if not math.isnan(target_values[ptr]):
                        future_target_values.append(target_values[ptr])
                        future_target_timestamps.append(str(timestamps[ptr]))
                        future_target_positions.append(len(future_indices) - 1)
                    ptr += 1

                if len(future_target_values) < self.forecast_window:
                    continue

                future_slice = feature_values[future_indices]
                future_mask = feature_masks[future_indices]
                future_ts = timestamps[future_indices]

                future_target_history_column = np.full((len(future_indices), 1), np.nan, dtype=np.float32)
                future_slice = np.concatenate([future_slice, future_target_history_column], axis=1)
                future_mask = np.concatenate([future_mask, np.isnan(future_target_history_column)], axis=1)

                target_slice = np.array(future_target_values, dtype=np.float32)
                target_mask = np.zeros_like(target_slice, dtype=bool)

                future_noisy, noise = self.add_noise_to_future(
                    future_slice,
                    future_mask,
                    future_ts,
                    history_ts[-1],
                    area_name,
                )

                if self.stats_accumulator:
                    self.stats_accumulator.update(history_slice)
                    self.stats_accumulator.update(future_slice)
                    if self.target_stats is not None:
                        self.target_stats.update(target_slice.reshape(-1, 1))

                samples_history.append(history_slice)
                samples_history_mask.append(history_mask)
                samples_history_timestamps.append([str(ts) for ts in history_ts])
                samples_future.append(future_noisy)
                samples_future_mask.append(future_mask)
                samples_future_noise.append(noise)
                samples_future_timestamps.append([str(ts) for ts in future_ts])
                samples_target.append(target_slice)
                samples_target_mask.append(target_mask)
                samples_target_timestamps.append(future_target_timestamps)
                samples_future_target_positions.append(future_target_positions)
                samples_climate.append(climate_value)

                record = {
                    "area": area_name,
                    "csv": csv_path.name,
                    "history_start": samples_history_timestamps[-1][0],
                    "history_end": samples_history_timestamps[-1][-1],
                    "future_start": future_target_timestamps[0],
                    "future_end": future_target_timestamps[-1],
                    "climate": climate_value,
                }
                metadata_records.append(record)

        if not samples_history:
            self.logger.info("Nessun campione valido per %s", area_name)
            return

        history_array = np.array(samples_history, dtype=object)
        history_mask_array = np.array(samples_history_mask, dtype=object)
        history_ts_array = np.array(samples_history_timestamps, dtype=object)
        future_array = np.array(samples_future, dtype=object)
        future_mask_array = np.array(samples_future_mask, dtype=object)
        future_noise_array = np.array(samples_future_noise, dtype=object)
        future_ts_array = np.array(samples_future_timestamps, dtype=object)
        target_array = np.stack(samples_target).astype(np.float32)
        target_mask_array = np.stack(samples_target_mask).astype(bool)
        target_ts_array = np.array(samples_target_timestamps, dtype=object)
        future_target_pos_array = np.array(samples_future_target_positions, dtype=np.int64)

        cache_path = self.cache_root / f"{area_name}.npz"
        np.savez_compressed(
            cache_path,
            history=history_array,
            history_mask=history_mask_array,
            history_timestamps=history_ts_array,
            future=future_array,
            future_mask=future_mask_array,
            future_noise=future_noise_array,
            future_timestamps=future_ts_array,
            target=target_array,
            target_mask=target_mask_array,
            future_target_positions=future_target_pos_array,
            target_timestamps=target_ts_array,
            climate=np.array(samples_climate),
        )

        metadata = {
            "area": area_name,
            "feature_names": self.feature_names,
            "target_column": self.target_column,
            "history_window": self.history_window,
            "forecast_window": self.forecast_window,
            "step": self.step,
            "noise_percentage": self.noise_percentage,
            "scaler_mode": self.scaler_mode,
            "min_samples_per_month": self.min_samples_per_month,
            "g5_target": self.g5_target,
            "metadata_records": metadata_records,
        }

        metadata_path = self.cache_root / f"{area_name}.json"
        with open(metadata_path, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

        self.logger.info("Creati %d campioni per %s", len(samples_history), area_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Generatore di cache con rumore mensile per area.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="timeSeries/global/train",
        help="Directory radice che contiene i CSV sorgente.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default="timeSeries/cache/train",
        help="Directory di destinazione per salvare le cache.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="timeSeries/dataSummary.csv",
        help="Percorso del file di metadata (dataSummary.csv).",
    )
    parser.add_argument(
        "--legend-path",
        type=str,
        default="timeSeries/csv_legend.xlsx",
        help="Percorso del file delle legende delle colonne.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="avg_NDVI_all",
        help="Nome della colonna target da prevedere.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=5,
        help="Numero di osservazioni Sentinel nel passato da includere.",
    )
    parser.add_argument(
        "--forecast-window",
        type=int,
        default=5,
        help="Numero di osservazioni Sentinel future da prevedere.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=6,
        help="Intervallo (in passi temporali) con cui campionare le finestre.",
    )
    parser.add_argument(
        "--noise-percentage",
        type=float,
        default=0.1,
        help="Ampiezza del rumore gaussiano relativo da applicare alle feature future.",
    )
    parser.add_argument(
        "--min-samples-per-month",
        type=int,
        default=100,
        help="Numero minimo di campioni mensili per usare la std locale dell'area.",
    )
    parser.add_argument(
        "--g5-target",
        type=float,
        default=2.0,
        help="Fattore di crescita desiderato all'ultimo step (g5).",
    )
    parser.add_argument(
        "--scaler-mode",
        type=str,
        default="min-max",
        choices=("min-max", "standardization", "standardization_arcsinh"),
        help="Tipologia di scaler da salvare insieme alla cache.",
    )
    parser.add_argument(
        "--add-temporal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aggiunge feature temporali derivate (disabilitabile con --no-add-temporal).",
    )
    parser.add_argument(
        "--fourier-harmonics",
        type=int,
        default=3,
        help="Numero di armoniche Fourier da calcolare per le feature temporali.",
    )
    parser.add_argument(
        "--include-hour",
        action="store_true",
        help="Include l'ora del giorno tra le feature temporali.",
    )
    parser.add_argument(
        "--include-day",
        action="store_true",
        help="Include il giorno della settimana tra le feature temporali.",
    )
    parser.add_argument(
        "--include-month",
        action="store_true",
        help="Include il mese tra le feature temporali.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per la generazione casuale del rumore.",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mostra barre di avanzamento (disabilitabile con --no-show-progress).",
    )
    parser.add_argument(
        "--group-mode",
        type=str,
        default="directory",
        choices=("directory", "pattern"),
        help="Modalità di raggruppamento dei CSV (per directory o via pattern nel nome).",
    )
    parser.add_argument(
        "--pattern-index",
        type=int,
        default=2,
        help="Indice del token nel nome file da usare come identificatore area (quando group-mode=pattern).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    cli_args = parse_args()
    builder = MonthlyNoiseCacheBuilder(**vars(cli_args))
    builder.build()
