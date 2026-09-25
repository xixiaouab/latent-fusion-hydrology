# Latent Fusion for Hydrological Forecasting

Fusing the spatial latents of **ORBIT-2** (a 126M-parameter climate downscaling
vision transformer) into a **frozen time-series hydrology foundation model**
(127M) through a trainable gated cross-attention adapter — so streamflow
forecasts can see the *spatial* structure of weather and land-surface state,
not just basin-averaged series.

Both backbones stay frozen; only the ~100M fusion adapter is trained.

## Headline results

Median NSE across basins, validation 2010–2012, **future weather fully
masked** (no future information reaches either model; spatial latents are
taken only from the 30 days before issue time).

| Downstream task | Baseline (TS model alone) | + ORBIT-2 frozen | + ORBIT-2 fine-tuned (top block) |
|---|---|---|---|
| Daily streamflow (CAMELS, 671 basins) | 0.402 | **0.559 (+0.156)** | 0.516 (+0.114) |
| Stream temperature | 0.873 | **0.908 (+0.035)** | 0.906 (+0.033) |
| Regulated (dam-controlled) basins | 0.886 | 0.891 (+0.005) | ≈ 0 (negative control, as predicted) |
| vs. National Water Model v2.1 | NWM retrospective scores **0.358** on the identical 384 evaluation windows — below even the masked baseline, despite being driven by observed weather | | |

Cross-task invariants:

1. **The harder the task, the larger the fusion gain** (streamflow ≫
   temperature ≫ dam-controlled, where the ceiling is set by human
   operations, not information).
2. **Fine-tuning ORBIT-2 adds no net gain over freezing it** — measured
   against architecture-matched frozen controls on two tasks (net +0.01 and
   −0.003). The pretrained spatial representation is already sufficient;
   compute is better spent on adapter capacity and longer latent windows
   (30-day window ≫ 10-day: monthly-scale spatial memory — snowpack, soil
   moisture — is a major value source).

## Method (three steps)

1. **Spatial encoding.** ORBIT-2 encodes each day's CONUS field (19
   variables × 180 × 360 at 10 arcmin) into 16,200 tokens × 1024. A basin's
   daily latent = area-weighted average of the tokens its polygon covers.
2. **Gated cross-attention adapter** (the only trained part):
   (a) 4 self-attention layers contextualize the 30-day latent sequence;
   (b) one query per forecast day (TS hidden state, 768→1024 projection,
   plus a day embedding) runs through 4 cross-attention blocks (16 heads,
   width 1024) over that sequence;
   (c) an output head produces per-day corrections, multiplied by a
   **zero-initialized tanh gate** and added to the backbone's quantile
   forecast — training starts exactly at the baseline (Flamingo-style).
3. **Loss.** Pinball (quantile) loss in arcsinh space, median weighted 3×;
   effective batch 128; DDP data parallelism.

## Data & variables

**From CAMELS (671 US basins)** — the time-series model's target and geometry:

| Content | Variables | Role |
|---|---|---|
| Observed daily discharge (`obsFlow`) | streamflow, cfs → mm/day (area-normalized) | forecast target + autoregressive history |
| Daymet basin-mean forcing | `prcp_mmday`, `srad_wm2`, `tmax_c`, `tmin_c`, `vp_pa` | covariates of the legacy 5-variable protocol |
| Basin geometry (`HCDN_nhru_final_671.shp`) | polygons, `AREA` (m²), 8-digit `hru_id` | ORBIT-2 token masks; flow normalization |

CAMELS static catchment attributes are not used.

**The 19 shared variables (headline protocol)** — basin-averaged for the
time-series model, gridded (19 × 180 × 360/day) for ORBIT-2; canonical
list at `src/extract_latents.py:77`:

| Group | Variables |
|---|---|
| Static (4) | `land_sea_mask`, `landcover`, `orography`, `lattitude` (archive's own spelling) |
| Temperature (4) | `2m_temperature`, `temperature_850/500/200` |
| Wind (6) | `u_component_of_wind_850/500/200`, `v_component_of_wind_850/500/200` |
| Humidity (3) | `specific_humidity_850/500/200` |
| Water (2) | `total_precipitation_24hr`, `volumetric_soil_water_layer_1` |

Both branches see the same 19 variables — basin-averaged vs. gridded — so
the comparison isolates spatial structure alone. Splits: train 1980–2009,
validation 2010–2012, test 2013–2014; 365-day calendar (leap years drop
Dec 31). Full details: `docs/data_and_parallelism.md`.

## Repository layout

```
src/
  fusion_train.py      # core: adapter, Store, DDP training loop, all ablation flags
  extract_latents.py   # ORBIT-2 latent extraction over the daily archive
  fixload.py           # robust loader for the TS backbone
  make_meta.py, transpose_latents.py, mean19.py   # dataset table builders
  extract_h7.py, transpose_h7.py    # blocks[0..6] cache for top-block fine-tuning
  make_wxstore.py      # normalized gridded-input store for full in-loop fine-tuning
  nwis_temp.py         # stream-temperature target download (NWIS 00010)
  exp4_eval.py, exp4_train.py       # regulated-basin (dam) experiments
  dump_valpairs.py, nwm_fetch.py    # NWM v2.1 comparison on identical windows
slurm/                 # Slurm launchers (fill in #SBATCH -A YOUR_PROJECT)
docs/protocol.md       # masking protocol, fairness rules, metric definition
docs/data_and_parallelism.md   # TS-model variables, the dataloader, time-alignment
                               # guarantees, and how DDP handles the two-model merge
                               # (with file:line pointers into the code)
```

## Setup

```bash
pip install -r requirements.txt
export FUSION_ROOT=/path/to/workdir        # holds dataset/, latents/, runs/, masks/
export TS_MODEL_DIR=/path/to/ts_model      # the time-series backbone package + weights/
export ERA5_DAYMET_DIR=/path/to/era5-daymet/10.0_arcmin
mkdir -p $FUSION_ROOT/logs                 # Slurm launchers write logs here
```

The pipeline expects, under `$FUSION_ROOT/dataset/`: `streamflow.npy`
(days × basins), `forcing19.npy` (days × basins × 19 basin-averaged
variables), transposed latents (basins × days × 1024), and `meta.json`
(built by `make_meta.py`). The calendar is 365-day (leap years drop
Dec 31), matching the Daymet archive.

## Running

```bash
# 1. extract ORBIT-2 latents (multi-rank; one year shard per rank)
python src/extract_latents.py ...
python src/transpose_latents.py            # (basin, day, dim) layout for training reads

# 2. train the fusion adapter (single node, 8 GPUs, DDP)
sbatch slurm/big_frontier.sh <cell-name> <mask-days> ...

# variants
sbatch slurm/temp_big.sh t10_big 10 1 0    # stream-temperature target
sbatch slurm/h7_cache.sh                   # then --orbit-topft {frozen,ft}
sbatch slurm/fullft_frontier.sh            # full 126M in-loop fine-tuning (16 nodes, bf16)

# NWM comparison (needs outbound internet, e.g. a data-transfer node)
python src/dump_valpairs.py                # freeze the exact evaluation windows
python src/nwm_fetch.py                    # selective read of NOAA's public zarr
```

## Honest notes

- Research code, validated end-to-end on OLCF Frontier (AMD MI250X, ROCm,
  torch ≥ 2.x). Baseline anchors reproduce bit-exact across all reported
  jobs; the full-fine-tuning path carries a built-in frozen-equivalence
  check (in-loop frozen == cached pipeline).
- The basin-mask builder script (CAMELS polygons → token weights, 6×6
  subpixel area weighting with centroid fallback) is archived with the
  validated mask data; see `docs/protocol.md` for the algorithm.
- Slurm launchers encode the exact configurations behind the reported
  numbers (walltimes sized for 2-hour partitions).

## License

MIT — see `LICENSE`.
