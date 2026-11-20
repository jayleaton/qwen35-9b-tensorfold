# HIP: ROG Ally X (gfx1103)

> **Pending device run.** The engine and the weights are checked on the machine's CPU; no number on this page comes
> from the GPU yet. The device runs wait on a reboot ([KNOWN-ISSUES](KNOWN-ISSUES.md#hip)).

| | |
| --- | --- |
| Machine | ROG Ally X: Ryzen Z1 Extreme, Radeon 780M (`gfx1103`), 24 GiB LPDDR5X with an 8 GiB UMA carve-out; Arch-based Linux, kernel 7.2, ROCm under `/opt/rocm` |
| Engine | upstream TensorFold `813ff4c` (149 commits after v1.2.10, an ancestor of the `upstream` config's `main`), **no patches**: [`config/hip/`](../config/hip/), tree `f70d190f1d1bd65575a53f7eb06838c7acf97cfc` |
| Weights | `q9-q4` from [CONVERT](CONVERT.md), the same portable MLX tree as the Spark and the Macs |

## Engine

Stock dense `qwen3_5` needs no patches on upstream's HIP path. On the pin above, CPU only:

- `zig build -Dhip=true test-qwen35-hip-host`: the family's host suite (config, loader, geometry, cache sizing) passes
  unpatched.
- `TF_QWEN_DIR=<tree> zig build test-qwen35`: the real-checkpoint loader test passes on the unpatched pin with the
  public Qwen3.5-4B MLX 4-bit tree (32 layers, 8 full attention, tied head, no MTP).
- The same test with `q9-q4` resolves every tensor with consistent shapes (32 layers, 8 full attention, untied head, no
  MTP, 3,683 MiB of projections) and passes. That run used the superset build described below, not the bare pin.

The official BF16 checkpoint itself fails the loader's mixed-dtype check (`linear_attn.norm.weight` is f32 among BF16
tensors); the converter writes those in bf16, which is why the converted tree is what every backend serves.

On the same GPU, a build that adds SenseNova-Loop's looped family to this pin (a strict superset for a stock
checkpoint) passed the stock Qwen3.5-4B device gate, `tf-qwen35-test check` 46 passed / 0 failed / 6 skipped, before
the GPU hang below.

## Weights

`scripts/convert.sh engine-metal Qwen3.5-9B q9-q4 4`, as in [CONVERT](CONVERT.md). The Ally X's tree has the same four
shard sha256 as the Spark's and the Macs' (5,950,219,464 bytes, 1,260 tensors, `mtp.*` left out, vision tower
carried):

```
2de5fd61485c13b58ab5186f0815610404c6a41d25afd84993a506c7d0f9a5db  model-00001.safetensors
f90abec9c4bd30b814d5a80b7adf0924592e1c5ca445b0a2c402825241fe54cb  model-00002.safetensors
e24d3cb49ac2671eb009249be9309e3ec2a7b37bbcdb047af3785969fc6a7d2a  model-00003.safetensors
febc767ef0fd7ed98ac28dbd81f0498b8cf57e98411989493b112fdb323db54c  model-00004.safetensors
```

## Build

`scripts/build-engine.sh` has no HIP target yet. Fetch the pin, then build with the device's ROCm (Arch keeps the
device libraries outside the default path):

```bash
scripts/fetch-engine.sh hip engine-hip
cd engine-hip
zig build -Dhip=true native-hip -Dhip-arch=gfx1103 -Dhipcc=/opt/rocm/bin/hipcc \
  -Dhip-device-lib=/opt/rocm/amdgcn/bitcode
```

There is no serve profile until the device run sets one.

## Device numbers

| measure | value |
| --- | --- |
| serve checklist (`/v1/models`, chat, streaming, tool calls, shutdown) | pending device run |
| decode, one chat, at 1K / 4K / 7-8K | pending device run |
| copy drafts against `"draft": false`: parity and speed | pending device run |
| peak device memory, largest window that fits | pending device run |
| prompt cache: resumed reply equals fresh | pending device run |
