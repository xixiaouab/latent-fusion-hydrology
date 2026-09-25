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
