"""Cross-attention fusion training: frozen TS-model × frozen ORBIT-2 latents.

v1 design (residual correction):
  q_final = q0_frozen + tanh(gate) * AdapterΔq        (gate init 0 → start == baseline)
  Adapter: TS-model future tokens (Q) cross-attend ORBIT-2 basin latents (K/V),
           2 gated blocks, zero-init output head, pinball loss in arcsinh space.

Run on a GPU node:
  python fusion_train.py --steps 3000 --batch 16
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

F = Path(os.environ.get("FUSION_ROOT", "."))
HYDRO = Path(os.environ.get("TS_MODEL_DIR", "ts_model"))
sys.path.insert(0, str(HYDRO))
from fixload import load_pipeline  # noqa: E402

CTX, HOR, N_Q = 730, 10, 21          # 2y daily context, 10-day horizon
LAT_PAST, LAT_FUT = 30, 10           # latent window rel. to origin
FORC = 5                              # [prcp, srad, tmax, tmin, vp]


class FusionAdapter(nn.Module):
    def __init__(self, d_h=768, d_l=1024, n_heads=8, n_blocks=2, per_day=False,
                 latent_layers=0, backbone_dim=768):
        super().__init__()
        self.per_day = per_day
        W = LAT_PAST + LAT_FUT
        self.q_proj = (nn.Linear(backbone_dim, d_h)
                       if d_h != backbone_dim else None)
        self.lat_enc = nn.ModuleList(
            nn.TransformerEncoderLayer(d_h, n_heads, dim_feedforward=4 * d_h,
                                       dropout=0.0, batch_first=True,
                                       norm_first=True)
            for _ in range(latent_layers))
        self.proj = nn.Linear(d_l, d_h)
        self.time_emb = nn.Parameter(torch.randn(1, W, d_h) * 0.02)
        if per_day:
            self.day_emb = nn.Parameter(torch.randn(1, HOR, d_h) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(d_h), "ln_kv": nn.LayerNorm(d_h),
                "attn": nn.MultiheadAttention(d_h, n_heads, batch_first=True),
                "ln_m": nn.LayerNorm(d_h),
                "mlp": nn.Sequential(nn.Linear(d_h, 4 * d_h), nn.GELU(),
                                     nn.Linear(4 * d_h, d_h)),
            }))
        self.out = nn.Linear(d_h, N_Q if per_day else N_Q * HOR)
        # Flamingo-style: ONLY the gate starts at zero. Zero-initing the output
        # head too creates a exact-zero-gradient deadlock (grad_gate ∝ out=0,
        # grad_out ∝ tanh(gate)=0) and nothing ever trains.
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, fut_tok, lat):
        if self.q_proj is not None:
            fut_tok = self.q_proj(fut_tok)
        kv = self.proj(lat.float()) + self.time_emb
        for lyr in self.lat_enc:
            kv = lyr(kv)
        if self.per_day:
            x = fut_tok.mean(1, keepdim=True).repeat(1, HOR, 1) + self.day_emb
        else:
            x = fut_tok
        for b in self.blocks:
            a, _ = b["attn"](b["ln_q"](x), b["ln_kv"](kv), b["ln_kv"](kv),
                             need_weights=False)
            x = x + a
            x = x + b["mlp"](b["ln_m"](x))
        if self.per_day:
            dq = self.out(x).permute(0, 2, 1)            # (B, N_Q, HOR)
        else:
            dq = self.out(x.mean(1)).view(-1, N_Q, HOR)
        return torch.tanh(self.gate) * dq


class TopBlockORBIT(nn.Module):
    """ORBIT-2's top ViT block + final norm, run on basin-local token sets.

    Weights come from the released checkpoint (topft_meta.pt); attention is
    over the basin's own patches only (3-20 tokens) rather than the full
    16,200-token grid — the localization approximation is controlled for by
    the 'frozen' mode cell."""

    def __init__(self, meta_path, d=1024, heads=16):
        super().__init__()
        self.h = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=True)
        self.proj = nn.Linear(d, d, bias=True)
        self.norm2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.normf = nn.LayerNorm(d)
        sd = torch.load(meta_path, map_location="cpu", weights_only=False)["top_sd"]
        m = {"norm1.weight": "blocks.7.norm1.weight", "norm1.bias": "blocks.7.norm1.bias",
             "qkv.weight": "blocks.7.attn.qkv.weight", "qkv.bias": "blocks.7.attn.qkv.bias",
             "proj.weight": "blocks.7.attn.proj.weight", "proj.bias": "blocks.7.attn.proj.bias",
             "norm2.weight": "blocks.7.norm2.weight", "norm2.bias": "blocks.7.norm2.bias",
             "fc1.weight": "blocks.7.mlp.fc1.weight", "fc1.bias": "blocks.7.mlp.fc1.bias",
             "fc2.weight": "blocks.7.mlp.fc2.weight", "fc2.bias": "blocks.7.mlp.fc2.bias",
             "normf.weight": "norm.weight", "normf.bias": "norm.bias"}
        own = self.state_dict()
        for k, src in m.items():
            assert sd[src].shape == own[k].shape, (k, sd[src].shape, own[k].shape)
            own[k] = sd[src]
        self.load_state_dict(own)

    def forward(self, tok, msk, wts):
        B, T, K, D = tok.shape
        x = tok.reshape(B * T, K, D).float()
        pad = ~msk.repeat_interleave(T, 0)                    # (B*T, K) True=pad
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B * T, K, 3, self.h, D // self.h)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)        # (B*T, h, K, hd)
        att = (q @ k.transpose(-2, -1)) * (D // self.h) ** -0.5
        att = att.masked_fill(pad[:, None, None, :], float("-inf"))
        att = att.softmax(-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B * T, K, D))
        x = x + self.fc2(nn.functional.gelu(self.fc1(self.norm2(x))))
        x = self.normf(x)
        w = (wts.repeat_interleave(T, 0)).unsqueeze(-1)       # zeros on pads
        return (x * w).sum(1).reshape(B, T, D)


class TopFTAdapter(nn.Module):
    """FusionAdapter with ORBIT-2's top block in front of the latent input."""

    def __init__(self, inner, top, trainable):
        super().__init__()
        self.inner = inner
        self.top = top
        if not trainable:
            self.top.requires_grad_(False)
        self.gate = inner.gate

    def forward(self, fut_tok, lat):
        tok, msk, wts = lat
        return self.inner(fut_tok, self.top(tok, msk, wts))


class FullFTOrbit(nn.Module):
    """The entire ORBIT-2 encoder in the training loop (all params trainable).

    Per-day forwards are gradient-checkpointed; basin pooling uses the same
    area weights as the offline extractor, so with frozen weights this path
    reproduces the cached-latent pipeline (global attention included)."""

    def __init__(self, chunk=2):
        super().__init__()
        import extract_latents as EX
        self.EX = EX
        H, Wg = 180, 360
        m = EX.Res_Slim_ViT(default_vars=EX.DEFAULT_VARS, img_size=(H, Wg),
                            in_channels=len(EX.IN_VARS), out_channels=1,
                            history=1, **EX.MODEL_KW)
        ck = torch.load(F / "ckpt/us_126m_precipitation.ckpt",
                        map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        msd = m.state_dict()
        for k in list(sd.keys()):
            if k not in msd:
                del sd[k]
            elif sd[k].shape != msd[k].shape:
                if k == "pos_embed":
                    EX.interpolate_pos_embed(m, sd, new_size=(H, Wg))
                else:
                    del sd[k]
        m.load_state_dict(sd, strict=False)
        m.data_config(18, (H, Wg), len(EX.IN_VARS), 1)
        # decoder-side modules never touched by forward_encoder: freeze them so
        # DDP does not wait for their gradients (and the optimizer skips them)
        for sub in ("head", "path2"):
            if hasattr(m, sub):
                getattr(m, sub).requires_grad_(False)
        self.orbit = m
        self.chunk = chunk
        mz = np.load(F / "masks/basin_patch_masks.npz", allow_pickle=True)
        nb = len(mz["basin_ids"])
        Wm = np.zeros((nb, 90 * 180), np.float32)
        for bi, (idx, w) in enumerate(zip(mz["patch_idx"], mz["patch_w"])):
            Wm[bi, np.asarray(idx, np.int64)] = np.asarray(w, np.float32)
        Wm = Wm / Wm.sum(1, keepdims=True)
        self.register_buffer("Wmat", torch.from_numpy(Wm), persistent=False)

    def day_tokens(self, x):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tok = self.orbit.forward_encoder(x.float(), self.EX.IN_VARS)
        return tok.float()

    def forward(self, wx, brow):
        B, Wn = wx.shape[0], wx.shape[1]
        flat = wx.reshape(B * Wn, *wx.shape[2:])
        toks = []
        for s in range(0, flat.shape[0], self.chunk):
            c = flat[s:s + self.chunk]
            if self.training and torch.is_grad_enabled():
                tok = torch.utils.checkpoint.checkpoint(
                    self.day_tokens, c, use_reentrant=False)
            else:
                tok = self.day_tokens(c)
            toks.append(tok)
        tok = torch.cat(toks, 0)                                  # (B*Wn, L, D)
        wrows = self.Wmat.index_select(0, brow)                   # (B, L)
        wrows = wrows.repeat_interleave(Wn, 0)                    # (B*Wn, L)
        lat = torch.einsum("bl,bld->bd", wrows, tok)
        return lat.reshape(B, Wn, -1)


class FullFTAdapter(nn.Module):
    def __init__(self, inner, orbit_mod):
        super().__init__()
        self.inner = inner
        self.orbit = orbit_mod
        self.gate = inner.gate

    def forward(self, fut_tok, lat):
        if isinstance(lat, (tuple, list)):
            wx, brow = lat
            lat = self.orbit(wx, brow)
        return self.inner(fut_tok, lat)


@torch.no_grad()
def fullft_panel_latents(orbit_mod, store, pairs, dev, day_bs=4):
    """Dedup panel latents: one forward per distinct day, pooled for 671 basins."""
    need = {}
    for bi, og in pairs:
        for d in range(og - LAT_PAST, og + LAT_FUT):
            need.setdefault(d, None)
    days = sorted(need)
    for s in range(0, len(days), day_bs):
        ds = days[s:s + day_bs]
        x = torch.from_numpy(np.ascontiguousarray(store.WX[ds])).to(dev)
        tok = orbit_mod.day_tokens(x)                             # (n, L, D)
        pooled = torch.einsum("nl,bld->bnd", orbit_mod.Wmat, tok)  # (nb, n, D)
        pn = pooled.half().cpu().numpy()          # (n_days, nb, D)
        for j, d in enumerate(ds):
            need[d] = pn[j]
    out = {}
    for bi, og in pairs:
        out[(bi, og)] = np.stack([need[d][bi]
                                  for d in range(og - LAT_PAST, og + LAT_FUT)])
    return out


def pinball(q, y, levels, wq=None):
    """q (B,21,H) , y (B,H) with NaN; arcsinh space; wq = per-quantile weights."""
    aq, ay = torch.arcsinh(q), torch.arcsinh(y).unsqueeze(1)
    m = torch.isfinite(ay)
    diff = ay - aq
    lv = levels.view(1, -1, 1)
    loss = torch.maximum(lv * diff, (lv - 1) * diff)
    if wq is not None:
        loss = loss * wq.view(1, -1, 1)
    return loss[m.expand_as(loss)].mean()


def nse(sim, obs):
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 5:
        return np.nan
    o, s = obs[m], sim[m]
    den = ((o - o.mean()) ** 2).sum()
    return 1 - ((s - o) ** 2).sum() / den if den > 1e-9 else np.nan


class Store:
    def __init__(self, target="streamflow.npy", flow_cov=False,
                 forcing="forcing.npy", topft=False):
        self.topft = topft
        d = F / "dataset"
        self.Q = np.load(d / "streamflow.npy")            # (days, nb) in RAM
        self.T = self.Q if target == "streamflow.npy" else np.load(d / target)
        self.X = np.load(d / forcing)                     # (days, nb, C) in RAM
        if flow_cov:
            # exp2 convention: streamflow becomes an input covariate series
            self.X = np.concatenate([self.X, self.Q[..., None]], axis=2)
        self.L = np.load(d / "latents_bm.npy", mmap_mode="r")  # (nb, days, 1024)
        self.meta = json.load(open(d / "meta.json"))
        self.years = self.meta["years"]
        # sparse targets (e.g. temperature): draw basins only where obs exist,
        # or rejection sampling starves
        tb = np.where(np.isfinite(self.T).sum(0) >= 1000)[0]
        self.tgt_basins = tb if 0 < len(tb) < self.T.shape[1] else None
        if topft:
            mt = torch.load(F / "h7_cache/topft_meta.pt", map_location="cpu",
                            weights_only=False)
            self.H7 = np.load(F / "h7_cache/h7_pm.npy", mmap_mode="r")
            self.bloc = mt["basin_local"]
            self.bw = mt["basin_w"]
        self.fullft = False
        self.lat_override = None
        if os.environ.get("FUSION_FULLFT") == "1":
            self.fullft = True
            self.WX = np.load(F / "wx_store.npy", mmap_mode="r")

    def year_range(self, y0, y1):
        i0 = self.years.index(y0) * 365
        i1 = (self.years.index(y1) + 1) * 365
        return i0, i1

    def sample_origins(self, y0, y1, n, rng, min_obs=5):
        i0, i1 = self.year_range(y0, y1)
        lo = max(i0, CTX + LAT_PAST)
        hi = i1 - HOR
        out = []
        tries = 0
        while len(out) < n and tries < n * 60:
            tries += 1
            og = int(rng.integers(lo, hi))
            if self.tgt_basins is None:
                bi = int(rng.integers(0, self.Q.shape[1]))
            else:
                bi = int(self.tgt_basins[rng.integers(0, len(self.tgt_basins))])
            fut = self.T[og:og + HOR, bi]
            ctx = self.T[og - CTX:og, bi]
            if np.isfinite(fut).sum() >= min_obs and np.isfinite(ctx).mean() >= 0.5:
                out.append((bi, og))
        return out

    def batch(self, pairs, dev):
        Bn = len(pairs)
        tgt = np.stack([self.T[og - CTX:og, bi] for bi, og in pairs])
        fx = np.stack([self.X[og - CTX:og, bi] for bi, og in pairs])
        ff = np.stack([self.X[og:og + HOR, bi] for bi, og in pairs])
        y = np.stack([self.T[og:og + HOR, bi] for bi, og in pairs])
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        if self.lat_override is not None:
            lat = t(np.stack([self.lat_override[(bi, og)] for bi, og in pairs]))
            return (t(tgt).float(), t(fx).float(), t(ff).float(),
                    t(y).float(), lat)
        if self.fullft:
            wx = np.stack([np.ascontiguousarray(
                self.WX[og - LAT_PAST:og + LAT_FUT]) for bi, og in pairs])
            brow = np.array([bi for bi, og in pairs], np.int64)
            lat = (t(wx), t(brow))
            return (t(tgt).float(), t(fx).float(), t(ff).float(),
                    t(y).float(), lat)
        if self.topft:
            Wn = LAT_PAST + LAT_FUT
            kmax = max(len(self.bloc[bi]) for bi, _ in pairs)
            tok = np.zeros((Bn, Wn, kmax, 1024), np.float16)
            msk = np.zeros((Bn, kmax), bool)
            wts = np.zeros((Bn, kmax), np.float32)
            for i, (bi, og) in enumerate(pairs):
                li = self.bloc[bi]
                sl = self.H7[li, og - LAT_PAST:og + LAT_FUT]   # (k, Wn, D)
                tok[i, :, :len(li)] = np.ascontiguousarray(sl).transpose(1, 0, 2)
                msk[i, :len(li)] = True
                wts[i, :len(li)] = self.bw[bi]
            lat = (t(tok), t(msk), t(wts))
        else:
            lat = t(np.stack([self.L[bi, og - LAT_PAST:og + LAT_FUT]
                              for bi, og in pairs]))
        return (t(tgt).float(), t(fx).float(), t(ff).float(),
                t(y).float(), lat)


def frozen_forward(pipe, tgt, fx, ff, dev):
    """Replicates pipeline._forward_batch but also returns future-token states."""
    model = pipe.model
    B = tgt.shape[0]
    nf = fx.shape[2]
    N = nf + 1
    ctx = pipe._pad_context(tgt, pipe.fc.context_length)
    fxp = pipe._pad_context(fx.permute(0, 2, 1).reshape(B * nf, -1),
                            pipe.fc.context_length).reshape(B, nf, -1)
    ctx_all = torch.cat([fxp, ctx.unsqueeze(1)], 1).reshape(B * N, -1)
    fut_all = torch.cat([ff.permute(0, 2, 1),
                         torch.full((B, 1, HOR), float("nan"), device=dev)],
                        1).reshape(B * N, HOR)
    gids = torch.arange(B, device=dev).repeat_interleave(N)
    n_out = math.ceil(HOR / pipe.fc.output_patch_size)
    out = model(context=ctx_all, future_covariates=fut_all, group_ids=gids,
                num_output_patches=n_out, output_attentions=True)
    sel = torch.arange(B, device=dev) * N + (N - 1)
    q0 = out.quantile_preds[sel]                                  # (B,21,HOR)
    fut_tok = out.encoder_hidden_states[sel][:, -n_out:, :]       # (B,n_out,768)
    return q0.float(), fut_tok.float()


class JointModel(nn.Module):
    """Backbone + adapter under one module so DDP buckets both gradients."""

    def __init__(self, pipe, adapter):
        super().__init__()
        self.bb = pipe.model
        self.adapter = adapter
        self._pipe_ref = [pipe]

    def forward(self, tgt, fx, ff, lat):
        q0, ftok = frozen_forward(self._pipe_ref[0], tgt, fx, ff, tgt.device)
        return q0, self.adapter(ftok, lat)


def day_curves(pipe, adapter, store, pairs, dev, bs=24, mask_days=0):
    """Median-across-basins NSE per lead day (per basin pooled over origins)."""
    obs, s0, s1 = {}, {}, {}
    for s in range(0, len(pairs), bs):
        chunk = pairs[s:s + bs]
        tgt, fx, ff, y, lat = store.batch(chunk, dev)
        if mask_days > 0:
            ff = ff.clone()
            ff[:, HOR - mask_days:, :] = float("nan")
        with torch.no_grad():
            q0, ftok = frozen_forward(pipe, tgt, fx, ff, dev)
            q = torch.clamp(q0 + adapter(ftok, lat), min=0.0)
        m0 = torch.clamp(q0, min=0.0)[:, N_Q // 2].cpu().numpy()
        m1 = q[:, N_Q // 2].cpu().numpy()
        yn = y.cpu().numpy()
        for i, (bi, og) in enumerate(chunk):
            obs.setdefault(bi, []).append(yn[i])
            s0.setdefault(bi, []).append(m0[i])
            s1.setdefault(bi, []).append(m1[i])
    c0, c1 = [], []
    for d in range(HOR):
        n0d, n1d = [], []
        for bi in obs:
            o = np.array([w[d] for w in obs[bi]])
            n0d.append(nse(np.array([w[d] for w in s0[bi]]), o))
            n1d.append(nse(np.array([w[d] for w in s1[bi]]), o))
        c0.append(float(np.nanmedian(n0d)))
        c1.append(float(np.nanmedian(n1d)))
    return c0, c1


def build_val_pairs(store, y0, y1, n_basins=64, per_basin=6):
    """Fixed panel: best-covered basins x evenly spread origins."""
    i0, i1 = store.year_range(y0, y1)
    cov = np.isfinite(store.T[i0:i1]).mean(0)
    lo = max(i0, CTX + LAT_PAST)
    pairs = []
    for bi in np.argsort(-cov):
        if cov[bi] < 0.7 or len(pairs) >= n_basins * per_basin:
            break
        got = 0
        for og in np.linspace(lo, i1 - HOR - 1, per_basin * 3).astype(int):
            fut = store.T[og:og + HOR, bi]
            ctx = store.T[og - CTX:og, bi]
            if np.isfinite(fut).sum() >= 8 and np.isfinite(ctx).mean() >= 0.5:
                pairs.append((int(bi), int(og)))
                got += 1
                if got >= per_basin:
                    break
    return pairs


def evaluate(pipe, adapter, store, pairs, dev, bs=24, nff=False, mask_days=None):
    """Hydrology-standard: per-basin pooled NSE over all windows, median across basins.

    mask_days=k hides the LAST k of the 10 future-forcing days (forecast-lead
    realism: near-term forecast known, far-term unknown). nff=True == k=10.
    """
    k = HOR if nff else int(mask_days or 0)
    acc = {}
    for s in range(0, len(pairs), bs):
        chunk = pairs[s:s + bs]
        tgt, fx, ff, y, lat = store.batch(chunk, dev)
        if k > 0:
            ff = ff.clone()
            ff[:, HOR - k:, :] = float("nan")
        with torch.no_grad():
            q0, ftok = frozen_forward(pipe, tgt, fx, ff, dev)
            q = torch.clamp(q0 + adapter(ftok, lat), min=0.0)
        # clamp base identically — otherwise fused gets a free "gain" that is
        # really just clipping the baseline's negative-flow artifacts
        med0 = torch.clamp(q0, min=0.0)[:, N_Q // 2].cpu().numpy()
        med1 = q[:, N_Q // 2].cpu().numpy()
        yn = y.cpu().numpy()
        for i, (bi, og) in enumerate(chunk):
            a = acc.setdefault(bi, ([], [], []))
            a[0].append(yn[i]); a[1].append(med0[i]); a[2].append(med1[i])
    n0, n1 = [], []
    for bi, (ys, s0, s1) in acc.items():
        o = np.concatenate(ys)
        n0.append(nse(np.concatenate(s0), o))
        n1.append(nse(np.concatenate(s1), o))
    return float(np.nanmedian(n0)), float(np.nanmedian(n1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=384)
    ap.add_argument("--per-day-queries", action="store_true")
    ap.add_argument("--median-weight", type=float, default=1.0)
    ap.add_argument("--ddp", action="store_true", help="multi-GPU data parallel (adapter grads all-reduced)")
    ap.add_argument("--no-future-forcing", action="store_true",
                    help="realistic protocol: future met unknown; latents are "
                         "the only future-weather signal (== --mask-days 10)")
    ap.add_argument("--mask-days", type=int, default=0,
                    help="hide the LAST k of 10 future-forcing days "
                         "(forecast-lead scarcity curve; 0=oracle, 10=fully masked)")
    ap.add_argument("--lat-past", type=int, default=30,
                    help="past days of ORBIT-2 latents visible to the adapter")
    ap.add_argument("--lat-fut", type=int, default=10,
                    help="future days of ORBIT-2 latents visible to the adapter "
                         "(0 = strictly pre-issue latents; 10 = includes horizon)")
    ap.add_argument("--forcing-npy", default="forcing.npy",
                    help="forcing table in dataset/ (forcing19.npy = 19-var basin means)")
    ap.add_argument("--unfreeze-ts", action="store_true",
                    help="fine-tune TS-model jointly with the adapter")
    ap.add_argument("--ts-lr-scale", type=float, default=0.1,
                    help="backbone LR = lr * this (unfreeze-ts only)")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-steps per optimizer step (keeps effective batch)")
    ap.add_argument("--d-hidden", type=int, default=768,
                    help="adapter width (768 reproduces the original 15M adapter)")
    ap.add_argument("--n-blocks", type=int, default=2,
                    help="cross-attention blocks")
    ap.add_argument("--latent-layers", type=int, default=0,
                    help="self-attention layers over the latent sequence (task "
                         "adaptation of the downscaling representation)")
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--orbit-topft", choices=["off", "frozen", "ft"], default="off",
                    help="run ORBIT-2's top transformer block in-loop on cached "
                         "pre-top-block features (basin-local attention): "
                         "frozen = localization control, ft = fine-tune its weights")
    ap.add_argument("--topft-lr-scale", type=float, default=0.1)
    ap.add_argument("--orbit-fullft", action="store_true",
                    help="full in-loop fine-tuning of ORBIT-2 (all 126M params); "
                         "reads wx_store.npy, gradient-checkpointed per-day forwards")
    ap.add_argument("--fullft-lr-scale", type=float, default=0.05)
    ap.add_argument("--fullft-chunk", type=int, default=2,
                    help="days per checkpointed ORBIT-2 forward chunk")
    ap.add_argument("--target-npy", default="streamflow.npy",
                    help="target table in dataset/ (e.g. temperature.npy)")
    ap.add_argument("--flow-as-covariate", action="store_true",
                    help="exp2 convention: streamflow joins the covariate set")
    ap.add_argument("--out", default=str(F / "runs/fusion_v1"))
    args = ap.parse_args()
    mk = HOR if args.no_future_forcing else max(0, min(HOR, args.mask_days))
    global LAT_FUT, LAT_PAST
    LAT_FUT = max(0, min(HOR, args.lat_fut))
    LAT_PAST = max(1, min(30, args.lat_past))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # DDP under srun: slurm vars carry rank; with --gpu-bind=closest each task
    # usually sees a single GCD, so the in-process device index is 0.
    rank, world, local = 0, 1, 0
    if args.ddp:
        rank = int(os.environ.get("SLURM_PROCID", 0))
        world = int(os.environ.get("SLURM_NTASKS", 1))
        local = int(os.environ.get("SLURM_LOCALID", 0)) % max(torch.cuda.device_count(), 1)
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", init_method="env://",
                                rank=rank, world_size=world)
    r0 = rank == 0

    def barrier():
        if args.ddp and world > 1:
            dist.barrier()

    dev = torch.device("cuda")
    rng = np.random.default_rng(42 + rank)

    if args.orbit_fullft:
        os.environ["FUSION_FULLFT"] = "1"
    store = Store(target=args.target_npy, flow_cov=args.flow_as_covariate,
                  forcing=args.forcing_npy, topft=args.orbit_topft != "off")
    if r0:
        print(f"TASK: target={args.target_npy} flow_cov={args.flow_as_covariate} "
              f"ncov={store.X.shape[2]}", flush=True)
    pipe = load_pipeline(HYDRO, device="cuda")
    pipe.model.eval()
    if not args.unfreeze_ts:
        for p in pipe.model.parameters():
            p.requires_grad_(False)
    levels = pipe.model.quantile_levels.to(dev).float()

    adapter = FusionAdapter(d_h=args.d_hidden, n_heads=args.n_heads,
                            n_blocks=args.n_blocks,
                            per_day=args.per_day_queries,
                            latent_layers=args.latent_layers).to(dev)
    if args.orbit_fullft:
        adapter = FullFTAdapter(adapter, FullFTOrbit(chunk=args.fullft_chunk)).to(dev)
        if r0:
            nfp = sum(p_.numel() for p_ in adapter.orbit.parameters())
            print(f"ORBIT_FULLFT: {nfp/1e6:.1f}M ORBIT-2 params in-loop "
                  f"(chunk={args.fullft_chunk})", flush=True)
    elif args.orbit_topft != "off":
        adapter = TopFTAdapter(adapter,
                               TopBlockORBIT(F / "h7_cache/topft_meta.pt"),
                               trainable=args.orbit_topft == "ft").to(dev)
        if r0:
            ntb = sum(p_.numel() for p_ in adapter.top.parameters())
            print(f"ORBIT_TOPFT mode={args.orbit_topft} top-block params "
                  f"{ntb/1e6:.1f}M (trainable={args.orbit_topft == 'ft'})", flush=True)
    model = None
    if args.unfreeze_ts:
        joint = JointModel(pipe, adapter)
        model = joint
        if args.ddp and world > 1:
            model = nn.parallel.DistributedDataParallel(joint, device_ids=[local])
        raw = adapter
    else:
        if args.ddp and world > 1:
            adapter = nn.parallel.DistributedDataParallel(
                adapter, device_ids=[local],
                find_unused_parameters=args.orbit_fullft)
        raw = adapter.module if isinstance(adapter, nn.parallel.DistributedDataParallel) else adapter
    n_par = sum(p.numel() for p in adapter.parameters())
    if r0:
        print(f"adapter params: {n_par/1e6:.2f}M "
              f"(per_day={args.per_day_queries}, median_w={args.median_weight}, "
              f"world={world})", flush=True)
    wq = torch.ones(N_Q, device=dev)
    wq[N_Q // 2] = args.median_weight
    wq = wq / wq.mean()
    if args.unfreeze_ts:
        opt = torch.optim.AdamW(
            [{"params": raw.parameters(), "lr": args.lr},
             {"params": pipe.model.parameters(),
              "lr": args.lr * args.ts_lr_scale}],
            weight_decay=1e-4)
        trainables = list(raw.parameters()) + list(pipe.model.parameters())
        if r0:
            nbb = sum(x.numel() for x in pipe.model.parameters())
            print(f"UNFREEZE_TS: backbone {nbb/1e6:.1f}M trainable at "
                  f"lr*{args.ts_lr_scale}, grad_accum={args.grad_accum}", flush=True)
    else:
        if args.orbit_fullft:
            ob_p = list(raw.orbit.parameters())
            ob_ids = {id(x) for x in ob_p}
            rest_p = [x for x in adapter.parameters()
                      if x.requires_grad and id(x) not in ob_ids]
            opt = torch.optim.AdamW(
                [{"params": rest_p, "lr": args.lr},
                 {"params": ob_p, "lr": args.lr * args.fullft_lr_scale}],
                weight_decay=1e-4)
            trainables = rest_p + ob_p
        elif args.orbit_topft == "ft":
            tb_p = list(adapter.top.parameters()) if hasattr(adapter, "top") \
                else list(adapter.module.top.parameters())
            tb_ids = {id(x) for x in tb_p}
            rest_p = [x for x in adapter.parameters()
                      if x.requires_grad and id(x) not in tb_ids]
            opt = torch.optim.AdamW(
                [{"params": rest_p, "lr": args.lr},
                 {"params": tb_p, "lr": args.lr * args.topft_lr_scale}],
                weight_decay=1e-4)
            trainables = rest_p + tb_p
        else:
            trainables = [x for x in adapter.parameters() if x.requires_grad]
            opt = torch.optim.AdamW(trainables, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, max(1, args.steps // args.grad_accum))

    # CAMELS lumped forcing/flow end 2014 -> splits confined to covered years
    tr = [1980, 2009]
    vl = [2010, 2012]   # test reserve: 2013-2014
    eval_bs = max(4, (24 * 6) // (store.X.shape[2] + 1))
    val_pairs = build_val_pairs(store, vl[0], vl[1])
    log = None
    if r0:
        print(f"val pairs: {len(val_pairs)} "
              f"({len(set(b for b, _ in val_pairs))} basins, per-basin pooled NSE)",
              flush=True)
        print(f"PROTOCOL: mask_days={mk}/10, latents t-{LAT_PAST}..t+{LAT_FUT} "
              f"({'past-only latents' if LAT_FUT == 0 else 'horizon-inclusive latents'})",
              flush=True)
        if args.orbit_fullft:
            store.lat_override = fullft_panel_latents(raw.orbit, store,
                                                      val_pairs, dev)
        b0, _ = evaluate(pipe, raw, store, val_pairs, dev, bs=eval_bs,
                         mask_days=mk)
        store.lat_override = None
        print(f"BASELINE frozen-TS-model val NSE(median): {b0:.4f}", flush=True)
        log = open(out / "train_log.jsonl", "a")
        log.write(json.dumps({"step": 0, "val_nse_base": b0,
                              "val_nse_fused": b0}) + "\n")
    barrier()
    best_nf = -1e9

    opt.zero_grad()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        pairs = store.sample_origins(tr[0], tr[1], args.batch, rng)
        if not pairs:
            continue
        tgt, fx, ff, y, lat = store.batch(pairs, dev)
        if mk > 0:
            ff = ff.clone()
            ff[:, HOR - mk:, :] = float("nan")
        if args.unfreeze_ts:
            q0, dq = model(tgt, fx, ff, lat)
        else:
            with torch.no_grad():
                q0, ftok = frozen_forward(pipe, tgt, fx, ff, dev)
            dq = adapter(ftok, lat)
        loss = pinball(torch.clamp(q0 + dq, min=0.0) + 1e-6, y, levels, wq)
        if not torch.isfinite(loss):
            opt.zero_grad()
            if r0 and step % 50 == 0:
                print(f"step {step}: non-finite loss skipped", flush=True)
            continue
        (loss / args.grad_accum).backward()
        if step % args.grad_accum == 0:
            nn.utils.clip_grad_norm_(trainables, 1.0)
            opt.step(); sched.step()
            opt.zero_grad()
        if step % 50 == 0 and rank == 0:
            print(f"step {step} loss {loss.item():.4f} gate {raw.gate.item():.3f} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            if r0:
                if args.orbit_fullft:
                    store.lat_override = fullft_panel_latents(
                        raw.orbit, store, val_pairs, dev)
                nb, nf = evaluate(pipe, raw, store, val_pairs, dev,
                                  bs=eval_bs, mask_days=mk)
                store.lat_override = None
                print(f"EVAL step {step}: base {nb:.4f} fused {nf:.4f} "
                      f"delta {nf-nb:+.4f}", flush=True)
                log.write(json.dumps({"step": step, "loss": float(loss.item()),
                                      "gate": float(raw.gate.item()),
                                      "val_nse_base": nb,
                                      "val_nse_fused": nf}) + "\n")
                log.flush()
                torch.save(raw.state_dict(), out / "adapter_last.pt")
                if nf > best_nf:
                    best_nf = nf
                    torch.save(raw.state_dict(), out / "adapter_best.pt")
                if args.unfreeze_ts:
                    torch.save(pipe.model.state_dict(),
                               out / "backbone_best.pt")
                    print(f"BEST checkpoint saved at step {step} (fused {nf:.4f})",
                          flush=True)
            barrier()
    if r0:
        best_p = out / "adapter_best.pt"
        if best_p.exists():
            raw.load_state_dict(torch.load(best_p, map_location=dev))
            bb_p = out / "backbone_best.pt"
            if bb_p.exists():
                pipe.model.load_state_dict(torch.load(bb_p, map_location=dev))
        big = build_val_pairs(store, vl[0], vl[1], n_basins=64, per_basin=24)
        if args.orbit_fullft:
            store.lat_override = fullft_panel_latents(raw.orbit, store, big, dev)
        c0, c1 = day_curves(pipe, raw, store, big, dev, bs=eval_bs,
                            mask_days=mk)
        store.lat_override = None
        print("DAYCURVE_BASE " + " ".join(f"{x:.4f}" for x in c0), flush=True)
        print("DAYCURVE_FUSED " + " ".join(f"{x:.4f}" for x in c1), flush=True)
        log.write(json.dumps({"day_curve_base": c0,
                              "day_curve_fused": c1}) + "\n")
        log.flush()
    barrier()
    if args.ddp:
        dist.destroy_process_group()
    if r0:
        print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
