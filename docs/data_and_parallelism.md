# Data, alignment, and distributed training

This note answers three questions precisely, with pointers into the code:
(1) what variables the time-series model consumes, (2) how the dataloader
is built and how the two models are guaranteed to read the *same time
period*, and (3) how distributed training works when two backbones and an
adapter are merged into one trainable system.

---

## 1. Variables consumed by the time-series model

The TS backbone receives, per basin:

| Input | Content | Where in code |
|---|---|---|
| Target history | past daily streamflow (mm/day, area-normalized) — autoregressive context of `CTX = 730` days | `src/fusion_train.py:29` (constant), `src/fusion_train.py:337` (slice) |
| Covariate history | basin-averaged meteorology, same 730-day window | `src/fusion_train.py:338` |
| Future covariates | the 10-day horizon window — **masked to NaN** under the no-future-weather protocol | `src/fusion_train.py:339` (slice), `src/fusion_train.py:549` + `src/fusion_train.py:462-468` (masking) |

Two covariate sets are supported, selected by `--forcing-npy`
(`src/fusion_train.py:508` region):

- **19-variable set** (`forcing19.npy`, the headline protocol): the exact
  `IN_VARS` list that ORBIT-2 itself is defined on — 4 static
  (`land_sea_mask`, `landcover`, `orography`, `lattitude` — the archive's
  own spelling) + 15 dynamic (`2m_temperature`; `temperature_850/500/200`;
  `u_/v_component_of_wind_850/500/200`; `specific_humidity_850/500/200`;
  `total_precipitation_24hr`; `volumetric_soil_water_layer_1`).
  Canonical definition: **`src/extract_latents.py:77`** (`IN_VARS`).
  The basin-averaged table is built by `src/mean19.py`.
- Legacy 5-variable Daymet set (`prcp, srad, tmax, tmin, vp`),
  `src/fusion_train.py:31`.

For the stream-temperature task, streamflow itself joins the covariates
(`--flow-as-covariate`): `src/fusion_train.py:289-291`.

### What comes from CAMELS, exactly

CAMELS (671 US basins) contributes three things to the pipeline:

| CAMELS content | Variable(s) | Role | Where |
|---|---|---|---|
| Observed daily discharge (`obsFlow`) | streamflow, converted cfs → **mm/day** using basin area | **forecast target** + autoregressive input history | `streamflow.npy` (days × 671); areas from `meta.json` |
| Daymet basin-mean forcing | `prcp_mmday`, `srad_wm2`, `tmax_c`, `tmin_c`, `vp_pa` (precipitation, shortwave radiation, max/min temperature, vapor pressure) | covariates of the **legacy 5-variable protocol** (early cells) | channel names recorded in `src/make_meta.py:17`; consumed via `forcing.npy` |
| Basin geometry (`HCDN_nhru_final_671.shp`) | polygons + `AREA` (m²), `hru_id` zero-filled to 8 digits | ORBIT-2 token masks; area normalization of flow | `src/make_meta.py:11-14`; mask algorithm in `docs/protocol.md` |

Under the **headline 19-variable protocol**, the Daymet covariates are
replaced by basin means of the same 19 ERA5-Daymet variables ORBIT-2
reads (built from the gridded archive by `src/mean19.py`), so the two
branches share one variable set and the comparison isolates *spatial
form* alone. CAMELS remains the source of truth for the target flow and
the basin geometry in every protocol. CAMELS static catchment attributes
(climate/soil/geology indices) are **not** used anywhere.

**The fairness point:** the spatial branch (ORBIT-2) encodes *the same 19
variables* as gridded CONUS fields (19×180×360/day). Both branches see
identical information content; the only difference is whether spatial
structure survives (grid) or is averaged away (basin mean).

### The time-series model's native variables: CAMELSH

The time-series backbone's own data format is **CAMELSH** (*A Large-Sample
Hourly Hydrometeorological Dataset and Attributes at Watershed-Scale for
CONUS*; Tran et al., *Scientific Data*, 2025): 9,008 CONUS basins,
1980–2024, hourly. The
model's public example basins are CAMELSH files, and its pipeline
documentation reports checks on CAMELSH basins. The model itself has no
fixed variable vocabulary: its pre-training script reads hourly and daily
tables, auto-detects every numeric column, and at each step picks one
channel at random as the target and treats the rest as known covariates.
So any column set can be supplied at inference time; the CAMELSH set below
is the one it is built around.

| Column | Meaning | Unit (NLDAS-2) |
|---|---|---|
| `Tair` | near-surface (2 m) air temperature | K (the model's example files store °C) |
| `Qair` | near-surface specific humidity | kg kg⁻¹ |
| `PSurf` | surface pressure | Pa |
| `Wind_E` | eastward wind (10 m) | m s⁻¹ |
| `Wind_N` | northward wind (10 m) | m s⁻¹ |
| `LWdown` | downward longwave radiation at the surface | W m⁻² |
| `SWdown` | downward shortwave radiation at the surface | W m⁻² |
| `Rainf` | total precipitation | kg m⁻² per hour (= mm h⁻¹) |
| `CRainf_frac` | fraction of precipitation that is convective | 0–1 |
| `CAPE` | convective available potential energy | J kg⁻¹ |
| `PotEvap` | potential evaporation | kg m⁻² per hour (= mm h⁻¹) |
| `Streamflow` | USGS observed discharge — the usual target | area-normalized mm/day in the model's streamflow examples |

The eleven forcing columns come from NLDAS-2 (1/8°, hourly), averaged over
each basin. Hourly USGS streamflow exists for 3,166 of the 9,008 basins.
CAMELSH also ships static catchment attributes (439 from GAGES-II, 195 from
HydroATLAS); the time-series model has **no static input**, so none of them
are used, here or in its own examples.

Sources: CAMELSH paper
(<https://www.nature.com/articles/s41597-025-05612-6>), data
(<https://zenodo.org/records/15413207>), code
(<https://github.com/vinhngoctran/CAMELSH>).

### What each downstream task fed the time-series model

| Task | Step | Target | Covariates (history + future window) | Data |
|---|---|---|---|---|
| ① Streamflow, 671 CAMELS basins | daily | streamflow, mm/day (CAMELS `obsFlow`) | headline: basin means of the 19 shared variables (`forcing19.npy`); legacy cells: 5 Daymet variables (`src/fusion_train.py:31`); backbone-fine-tuning cells: the 15 dynamic variables (`forcing15.npy`; the 4 statics destabilize backbone fine-tuning) or the 5 Daymet ones | CAMELS + ERA5-Daymet archive |
| ② Stream temperature | daily | water temperature, °C (USGS NWIS parameter 00010) | 5 Daymet variables + streamflow (`--flow-as-covariate`, `src/fusion_train.py:289`); fine-tuned-backbone cells: 15 dynamic + streamflow (`slurm/temp_big.sh:22`) | CAMELS + NWIS |
| ③ Sub-hourly, Luxembourg (CAMELS-LUX basin 40) | 15 min | discharge `Q`, m³/s | `Precip`, `AirTemp` | the time-series model's example basin file |
| ④ Regulated basin (Carson River, USGS 10312150) | daily | streamflow, mm/day | the 11 CAMELSH forcing columns (`src/exp4_train.py:34`); + `DamOutflow` (reservoir release from ResOpsUS, m³/s → mm/day) in the "with release" cells | the time-series model's example basin file (CAMELSH format) |
| ⑤ vs. National Water Model | daily | same as ① | same as ① — identical 384 evaluation windows | CAMELS + NWM v2.1 retrospective |

So the CAMELSH columns reach our experiments directly only in task ④. In the
CAMELS tasks the backbone is fed daily basin means from CAMELS/Daymet or
from the ERA5-Daymet archive shared with ORBIT-2, which is what keeps the
two branches' information content identical.

---

## 2. The dataloader (`Store`) — and why alignment is by construction

There is deliberately **no `torch.utils.data.DataLoader`**. Training draws
random `(basin, origin-day)` pairs from tables that live in RAM or
memory-map, and assembles a batch in microseconds. The whole loader is one
class: **`Store`, `src/fusion_train.py:281-370`**.

### Tables (all indexed by ONE global day axis)

| Table | Shape | Residence | Loaded at |
|---|---|---|---|
| `streamflow.npy` | (days, basins) | RAM | `src/fusion_train.py:286` |
| `forcing19.npy` | (days, basins, 19) | RAM | `src/fusion_train.py:288` |
| `latents_bm.npy` | **(basins, days, 1024)** | mmap | `src/fusion_train.py:292` |
| `h7_pm.npy` (top-block variant) | (patches, days, 1024) | mmap | `src/fusion_train.py:302` |
| `wx_store.npy` (full-FT variant) | (days, 19, 180, 360) | mmap | `src/fusion_train.py:309` |

The day axis is identical across every table: **day 0 = 1980-01-01,
365-day years, leap years drop Dec 31** (the Daymet convention of the
gridded archive). Year→index conversion is exactly
`index = (year - 1980) * 365`: `src/fusion_train.py:311-314`
(`year_range`). Every new data source (e.g. NWIS temperature,
`src/nwis_temp.py`) is forced onto the same calendar at table-build time.

Note the latent table is stored **basin-major** (`(basins, days, dim)`,
built by `src/transpose_latents.py`): training reads "one basin, a
contiguous day window", so the layout must match the access pattern —
day-major latents caused Lustre page-fault crawl.

### One sample = one `(bi, og)` pair drives *every* slice

`Store.batch()` (`src/fusion_train.py:335-370`) takes a list of
`(basin bi, origin day og)` pairs. For each pair, **all five tensors are
sliced from the same two integers**:

```
target history  T[og-730 : og,        bi]   line 337
covariate hist  X[og-730 : og,        bi]   line 338
future covs     X[og     : og+10,     bi]   line 339   (masked later per protocol)
labels          T[og     : og+10,     bi]   line 340
spatial latents L[bi, og-30 : og+LAT_FUT]   lines 367-368  (LAT_FUT=0 in the strict protocol)
```

There are no separate loaders, iterators, or clocks for the two models —
**misalignment is structurally impossible** because there is exactly one
source of time indices. The top-block variant (lines 353-365) and the
full-FT variant (lines 346-352) slice their caches with the *same* `og`
arithmetic.

Origin sampling (`sample_origins`, `src/fusion_train.py:316-333`) rejects
pairs with insufficient observations; the fixed validation panel
(`build_val_pairs`, `src/fusion_train.py:441`) is deterministic given the
tables, which is what lets the NWM comparison replay the identical
windows offline (`src/dump_valpairs.py`).

### Three runtime checks that would expose any alignment bug

1. **Bit-exact baseline anchor** — the frozen baseline (0.4023 streamflow /
   0.8730 temperature) must reproduce to the last digit in every job; it
   depends only on the TS-side slices, so any calendar or index drift
   changes it immediately.
2. **Zero-initialized gate** (`src/fusion_train.py:34` region) — training
   starts exactly at the baseline; a misaligned latent branch shows up as
   an immediate departure at step 0.
3. **Frozen-equivalence identity** — with ORBIT-2 frozen, the in-loop
   full-FT path must equal the cached-latent pipeline mathematically
   (`FullFTAdapter`, `src/fusion_train.py:223`); this cross-validates the
   two data routes end to end, including normalization
   (`src/make_wxstore.py` must match the extractor byte-for-byte).

---

## 3. Distributed training — "how do you distribute two models?"

Short answer: **pure data parallelism (PyTorch DDP); the two backbones and
the adapter are packed into ONE `nn.Module` wrapper, and DDP wraps that
single module.** No model/tensor/pipeline parallelism is used or needed —
the whole system (127M TS + 126M ORBIT-2 + ~100M adapter ≈ 350M params)
fits on a single GPU; what is expensive is samples, not memory.

### Wiring (Slurm → ranks → GPUs)

`src/fusion_train.py:559-569`:

```
rank  = $SLURM_PROCID          # one process per GPU
world = $SLURM_NTASKS
local = $SLURM_LOCALID % visible GPUs     → torch.cuda.set_device(local)
init_process_group(backend=NCCL/RCCL, MASTER_ADDR/PORT from the launcher)
```

Every rank holds a full replica of all three networks and its own `Store`
(read-only tables — no sampler coordination needed; each rank draws its
own random pairs). Per step, each GPU computes forward/backward on its
shard; DDP all-reduces gradients once; all replicas stay identical.
Effective batch = per-GPU batch × world × grad-accum = **128 in every
reported cell** (e.g. 16×8, 2×8×8, 1×128).

**Data sharding without a `DistributedSampler`:** since there is no
epoch-over-a-dataset (training draws random `(basin, origin)` pairs from
a huge index space), sharding reduces to giving each rank an independent
random stream — `rng = np.random.default_rng(42 + rank)`,
`src/fusion_train.py:572`. Ranks therefore sample *different* pairs with
probability ≈ 1 (collisions are harmless — a duplicate pair is just an
i.i.d. redraw), which is what makes the effective batch genuinely 128
rather than 8 copies of 16. Evaluation, by contrast, runs on the fixed
deterministic panel on rank 0 only, so reported numbers never depend on
the rank layout.

### The "merge" question: one wrapper module, not two DDPs

Wrapping two networks in two separate DDP objects creates two independent
synchronization streams. Instead, whatever needs gradients is composed
into a single module, and DDP wraps that:

| Configuration | Trainable content | Wrapper (file:line) | DDP wrap |
|---|---|---|---|
| Frozen × frozen (champion) | adapter only | — | `src/fusion_train.py:615` |
| Fine-tune TS backbone | TS + adapter | `JointModel`, `src/fusion_train.py:396` | `src/fusion_train.py:611` |
| Fine-tune ORBIT-2 top block | top block + adapter | `TopFTAdapter`, `src/fusion_train.py:139` | `src/fusion_train.py:615` |
| Fine-tune ORBIT-2 fully (126M in-loop) | ORBIT-2 + adapter | `FullFTAdapter`, `src/fusion_train.py:223` | `src/fusion_train.py:615-617` |

Details that matter in practice:

- **Frozen parameters ride along for free.** DDP only registers
  parameters with `requires_grad=True` for reduction. In the champion
  configuration both backbones are frozen (`requires_grad_(False)`), so
  each rank carries identical read-only replicas and only the ~100M
  adapter is synchronized. The frozen cells don't even instantiate
  ORBIT-2 — its latents are precomputed to disk and mmap-read.
- **Different learning rates are the optimizer's job, not DDP's** —
  parameter groups give the adapter 2e-4 and the fine-tuned backbone
  ×0.05–0.1 of that.
- **`find_unused_parameters=True` for encoder-only use**
  (`src/fusion_train.py:617`): we call only ORBIT-2's `forward_encoder`,
  so its decoder parameters never receive gradients; without this flag
  DDP deadlocks waiting for them ("Expected to have finished reduction").
- **Rank-0 discipline**: evaluation, logging, and checkpointing happen on
  rank 0 only, with `dist.barrier()` fences
  (`src/fusion_train.py:689,746,765`); checkpoints save
  `wrapper.module.state_dict()` (the *raw* module) so keys carry no
  `module.` prefix — and the optimizer must reference `raw.orbit`, not
  the DDP-wrapped attribute path.
- **Scale-out is free**: the same script runs 1 node × 8 GPUs (frozen
  cells) and 16 nodes × 128 GPUs (full in-loop fine-tuning, bf16
  autocast + per-day gradient checkpointing) with no code change — only
  the launcher's node count differs (`slurm/fullft_frontier.sh`).

### Why not model parallelism?

The compute bottleneck in the full-FT cell is ORBIT-2's quadratic
attention over 16,200 tokens per daily image (~84 TF per image at fp32) —
a per-sample cost, not a memory-capacity problem. Data parallelism
attacks exactly that axis; splitting a 350M-parameter model across GPUs
would add communication for zero benefit.
