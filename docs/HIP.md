# HIP: ROG Ally X (gfx1103)

> **Keep GTT below the 8 GiB carve-out.** Boot with `ttm.pages_limit=1966080 ttm.page_pool_size=1966080` (7.5 GiB).
> With GTT above the carve-out (`4194304`, 15.3 GiB) the engine budgets from host memory, places its buffers in GTT, and
> the GPU hangs in prefill ([KNOWN-ISSUES](KNOWN-ISSUES.md#hip)).

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
checkpoint: without `loop_settings` every layer runs once and none of the loop code engages) passed the stock
Qwen3.5-4B device gate, `tf-qwen35-test check` 46 passed / 0 failed / 6 skipped. The device numbers below come from
that build.

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

`scripts/serve.sh` has no HIP profile yet. The device run served with:

```bash
tensorfold-native serve q9-q4 --context 8192 --parallel 1 --prompt-cache-gib 0 --no-update-check
```

It admits the 4.69 GiB model, one stream at 0.30 GiB for 8,192 tokens and a 1.0 GiB reserve, and loads in 14-16 s.

## Device numbers

2026-11-20, ROCm 7.2.4, greedy, 256-token replies unless noted.

| measure | value |
| --- | --- |
| decode, short prompt (21 tokens), drafts off | **19.0 tok/s**, TTFT 0.25 s |
| decode, 4.4K prompt (128-token replies) | **17.8 tok/s**, TTFT 11.7-12.3 s (about 370 tok/s prefill) |
| decode, 7.3K prompt | **17.1 tok/s**, TTFT 20.0-21.3 s |
| copy drafts against `"draft": false` | replies identical on every pair (short, 7.3K, re-quote, code edit, chat); re-quote **57.0** against 18.2 tok/s, code edit **53.3** against 18.2; plain chat unchanged (18.5 against 18.7) |
| peak device memory | 6.3 GiB at 8,192 tokens with no prompt cache; **7.4 GiB** with `--prompt-cache-gib 1` |
| prompt cache: resumed reply equals fresh | identical, twice (5K prompt): cold TTFT 13.4 s, resumed **0.30 s** |
| serve checklist | `/v1/models`, chat with thinking on, a streamed reply to `[DONE]`, tool calls with `auto` and `required` (both `finish_reason: tool_calls`), SIGINT drains and exits |
| HumanEval+ 164, think 1024, cap 1536 | **87.8** (144/164) against 89.6 for the stock 4-bit reference |
| BFCL held-out 150-row subset, think 512 | **80.0** (120/150) against 82.4 for the stock 4-bit reference on all 450 rows |

The two quality rows served with `--context 4096 --parallel 4 --thinking-reserve 2112 --prompt-cache-gib 0`. The stock
9B decodes about 1.7× faster than SenseNova-Loop-9B on the same GPU (10.8 tok/s at 1K), which runs 56 layer executions
a token against 32.
