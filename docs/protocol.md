# Evaluation protocol

## Masked-future ("no-future-weather") protocol

- The forecast horizon is 10 days. All dynamic covariates for the horizon
  are masked (`NaN`) for the time-series backbone: neither model sees any
  future weather.
- ORBIT-2 latents enter the adapter only from the **30 days before the
  issue date** (`--lat-fut 0`, `LAT_PAST = 30`). No future-dated spatial
  field reaches the input.
- The open-book control (`--mask-days 0`) gives both sides the full future
  window and measures redundancy: for streamflow the latent margin turns
  negative (gridded and basin-averaged futures are redundant); for stream
  temperature it stays positive (+0.0125) — spatial temperature structure
  carries information basin averages destroy.

## Fairness rules

1. Every cell's adapter is trained **from scratch** — no reuse of an
   adapter trained under a different information regime.
2. Effective batch size is 128 in every cell (per-GPU batch × GPUs × grad
   accumulation).
3. Fine-tuned-ORBIT-2 cells are compared against **architecture-matched
   frozen controls** (same localized attention context, same window), so
   localization cost and fine-tuning benefit are separated.
4. Baseline anchors: the frozen baseline must reproduce bit-exact across
   jobs (streamflow 0.4023; temperature 0.8730 masked / 0.9650 open).
   A cell whose baseline drifts is discarded as mis-configured.

## Metric

Per basin, the six 10-day forecast windows (evenly spread origins,
validation years 2010–2012) are concatenated and a single NSE is computed
against observations; the reported number is the **median across the
64-basin panel** (best-covered basins, coverage ≥ 0.7). `NaN` observations
are excluded pointwise. The same windows, observations, and metric are
applied verbatim to the NWM v2.1 retrospective comparison
(`dump_valpairs.py` freezes the panel; `nwm_fetch.py` evaluates NWM on it).

## Calendar

365-day years throughout (leap years drop Dec 31), matching the Daymet
convention of the gridded archive. All tables share one global day index
anchored at 1980-01-01; a single (basin, issue-date) pair drives both the
time-series windows and the latent windows — alignment by construction.

## Basin masks

Basin polygon → token weights on the 90 × 180 patch grid (patch size 2 on
the 180 × 360 field): each polygon is rasterized at 6×6 subpixel
resolution inside every overlapping token cell to get fractional area
weights; basins smaller than one cell fall back to their centroid token.
The median CAMELS basin covers ~3 tokens. Validated masks (671/671 basins)
are archived with the data; the builder script operates on the CAMELS
`HCDN_nhru_final_671` shapefile (NAD83, `hru_id` zero-filled to 8 digits).

## Sub-hourly task (Luxembourg)

- **Data.** One CAMELS-LUX basin (no. 40): 15-minute discharge `Q` (m³/s)
  with `Precip` and `AirTemp`, 2020-01-01 to 2021-11-01 (the time-series
  model's sub-hourly example file; path via `LUX_CSV` or `TS_MODEL_DIR`).
- **Forecast.** 96 steps (24 h) from 8,760 steps (~91 days) of context, as in
  the model's own example.
- **Splits.** Training origins: every 15-minute step from 2020-04-01 whose
  24 h target ends before 2020-11-01. Validation: 30 origins spread over
  Nov–Dec 2020, used only for checkpoint selection. Test: the example
  notebook's own 30 origins across 2021, reproduced exactly.
- **Masking.** K = the number of trailing horizon steps whose future
  covariates are hidden. K = 96 is fully blind (the headline setting); K = 0
  is open-book, the example's setting.
- **Latents.** ORBIT-2's *global* fine-tuned checkpoint (23 inputs, ERA5
  0.25°) encodes a 128 × 128 window (32° × 32°, 36.25–68°N, 12°W–19.75°E) of
  daily ERA5 aggregates. The basin latent is the mean of the 3 × 3 token box
  over Luxembourg. The adapter sees the 30 calendar days *strictly before*
  the issue day; the issue day itself is excluded because its daily
  aggregate contains hours after the issue time.
- **Metric.** Pooled NSE over the 30 test forecasts × 96 steps. The
  open-book baseline reproduces the example's own anchor: median per-lead
  NSE 0.9455 against the published 0.946.
- **Global-checkpoint conventions** (each asserted in `extract_lux.py`):
  the static layers are 720 × 1440, south-up (row 0 = −90°), longitude
  0–359.75°, while ARCO-ERA5 is north-up, so the window is flipped. The
  model re-interpolates its position embeddings on every forward assuming a
  2:1 grid, which fails on a square window; the model is therefore built at
  the checkpoint's own 92 × 184-token grid and that interpolation is
  replaced by a centred crop (the embeddings are fixed sin-cos and the window
  is already at the native 0.25°). The model's code calls `torch.distributed`
  even on one GPU, so a one-rank gloo group is initialized. Precipitation is
  kept in m/day, the unit of the checkpoint's own statistics.
- **Multi-node binding.** With `--gpu-bind=closest`, ranks 6 and 7 of each
  node landed on the same GCD (RCCL "Duplicate GPU detected");
  `slurm/exp3_fullft.sh` requests `--gpus-per-node=8` without binding and
  each rank selects `cuda:SLURM_LOCALID`.
