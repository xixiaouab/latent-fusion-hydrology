"""Shared loader: from_pretrained + manual safetensors overwrite + checksum.

Written for transformers 5.x hazards (post-load re-init, meta-device
non-persistent buffers); on 4.x these are no-ops but the verification and
explicit RoPE rebuild remain a free integrity guarantee.
"""
from pathlib import Path

import torch
from safetensors.torch import load_file

CHECKSUM = 3558566.89


def load_pipeline(repo, device="cuda"):
    import sys
    sys.path.insert(0, str(repo))
    from src.pipeline import HydroORBITPipeline
    from src.layers import RoPE

    pipe = HydroORBITPipeline.from_pretrained(str(Path(repo) / "weights"),
                                              device=torch.device(device))
    sd = load_file(str(Path(repo) / "weights/model.safetensors"))
    pipe.model.load_state_dict(sd, strict=True)

    dev = next(pipe.model.parameters()).device
    n_rope = 0
    for m in pipe.model.modules():
        if isinstance(m, RoPE):
            inv = 1.0 / (m.base ** (
                torch.arange(0, m.dim, 2, dtype=torch.int64).float() / m.dim))
            m.register_buffer("inv_freq", inv.to(dev), persistent=False)
            n_rope += 1
            assert torch.isfinite(m.inv_freq).all() and m.inv_freq.max() <= 1.0

    pipe.model.to(pipe.device)
    cs = sum(v.double().abs().sum().item()
             for v in pipe.model.state_dict().values()
             if v.is_floating_point())
    assert abs(cs - CHECKSUM) < 1.0, f"weights corrupt: {cs:.2f}"
    print(f"fixload: weights_checksum={cs:.2f} OK, {n_rope} RoPE buffers rebuilt")
    return pipe
