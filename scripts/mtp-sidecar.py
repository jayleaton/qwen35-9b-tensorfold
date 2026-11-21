"""Write a Qwen3.5 checkpoint's MTP head as `mtp.safetensors`, a side file the engine reads beside a converted tree.

The converter drops `mtp.*`, so the tree's shards stay as they are; the engine's checkpoint reader takes any
`mtp*.safetensors` in the model directory as the head. Projections are quantized with the converter's own affine
routine at the tree's bits and group, and the head's RMSNorms get the converter's `1 + w` fold.

  python3 scripts/mtp-sidecar.py --source BF16_DIR --converter ENGINE/tools/sensenova_loop_mlx.py --out FILE \
      [--bits 4] [--group 64]
"""
import argparse
import hashlib
import importlib.util
import json
import os

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

PROJECTIONS = ("mtp.fc.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
               "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
NORMS = ("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight",
         "input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight")


def converter(path):
    spec = importlib.util.spec_from_file_location("sensenova_loop_mlx", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--converter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--rows", type=int, default=1024)
    a = ap.parse_args()
    conv = converter(a.converter)
    weight_map = json.load(open(os.path.join(a.source, "model.safetensors.index.json")))["weight_map"]
    names = sorted(n for n in weight_map if n.startswith("mtp."))
    if not names:
        raise SystemExit(f"{a.source} has no mtp.* tensors")
    out, handles = {}, {}
    try:
        for name in names:
            f = weight_map[name]
            if f not in handles:
                handles[f] = safe_open(os.path.join(a.source, f), "pt")
            t = handles[f].get_tensor(name).contiguous()
            if name.endswith(PROJECTIONS):
                n, k = t.shape
                assert n % 32 == 0 and k % a.group == 0, (name, tuple(t.shape))
                words = np.zeros((n, k * a.bits // 32), dtype=np.uint32)
                scales = np.zeros((n, k // a.group), dtype=np.uint16)
                biases = np.zeros((n, k // a.group), dtype=np.uint16)
                for r0 in range(0, n, a.rows):
                    r1 = min(n, r0 + a.rows)
                    w, s, b = conv.quantize_rows(t[r0:r1].float().numpy(), a.bits, a.group)
                    words[r0:r1], scales[r0:r1], biases[r0:r1] = w, s, b
                base = name[: -len(".weight")]
                out[f"{base}.weight"] = torch.from_numpy(words)
                out[f"{base}.scales"] = torch.from_numpy(scales.view(np.int16)).view(torch.bfloat16)
                out[f"{base}.biases"] = torch.from_numpy(biases.view(np.int16)).view(torch.bfloat16)
            elif name.endswith(NORMS):
                out[name] = (t.float() + 1.0).to(torch.bfloat16).contiguous()
            else:
                raise SystemExit(f"unexpected MTP tensor {name} {tuple(t.shape)}")
    finally:
        for h in handles.values():
            h.__exit__(None, None, None)
    meta = {"format": "pt", "quantization": json.dumps({"bits": a.bits, "group_size": a.group, "mode": "affine"})}
    save_file(out, a.out, metadata=meta)
    digest = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(json.dumps({"out": a.out, "tensors": len(out), "source_tensors": len(names),
                      "bytes": os.path.getsize(a.out), "sha256": digest}))


if __name__ == "__main__":
    main()
