# RTX 16 GB: an EXL3 pack on the CUDA engine

> **Tested on an RTX 5070 Ti 16 GB under WSL2**, with the Windows desktop on the same card and in use. Other 16 GB cards
> and native Windows are untested.

| | |
| --- | --- |
| Engine | upstream TensorFold `038e4a278` + [`patches/cuda/`](../patches/cuda/) `0001`-`0022` + [`patches/rtx/`](../patches/rtx/) `0001`-`0003` |
| Engine tree | `64ba7d4085cc4742c127af91fa196218b189bdbf` ([`config/rtx/`](../config/rtx/)), checked 2026-11-20 |
| Card | RTX 5070 Ti 16,303 MiB (Blackwell, `sm_120`), Windows driver 621.14; the desktop held 1.3-1.8 GiB of it |
| Host | Ryzen 9 9950X, 61.6 GiB RAM, Windows 11; WSL2 Ubuntu 24.04 with 30 GiB and 32 threads |
| Toolkit | nvcc 13.0.88 from NVIDIA's redist tarballs (no sudo), Zig 0.17.0, xgrammar v0.2.8 |
| Weights | `st-6.0`: EXL3 6.0 bpw from the BF16 checkpoint ([quantize](#quantize)), 8.28 GB |

`patches/rtx/` `0001`-`0002` are the SenseNova recipe's EXL3 loader and `cuda.zig` cleanup; for a stock pack the loop
and mHC parts stay idle. `0003` is a tools-only change: `tools/exl3_attach_sensenova.py attach --method` records how
the pack was calibrated, so a stock pack that exllamav3 calibrated itself does not claim SenseNova's Hessian method.

## Results on the RTX 5070 Ti

`st-6.0`, served with `config/rtx/serve.args`. Greedy, 256 tokens with `ignore_eos`, thinking off, Python
standard-library source as the long prompt. The desktop was in use throughout (GPU 5-21% busy before each stage), so
speeds are for a loaded card. Every drafted reply equals its `"draft": false` reply, and every restored reply equals
its cold one.

| check | result |
| --- | --- |
| load | ready in 10-13 s from ext4; `model 8.28 GiB; 1 stream at once, 4.23 GiB each at a 262144-token window (fp8 caches)` |
| largest fp8-KV window | **262,144**, the model's whole window, with 1.2 GiB of admission room; a 227,182-token prompt served |
| engine device peak | **12.49 GiB** (load, every probe, the near-full prompt); 12.93 GiB at `--parallel 4 --context 65536` |
| API | `/v1/models`, chat with thinking (`391`), streaming (`[DONE]`), tool calls `auto` and `required`: pass |
| decode at 4,037 / 30,209 tokens | **48-50 / 43-44 tok/s**, drafted and plain, 3 runs each; one reply sha per prompt |
| decode at 227,182 tokens | 28.0 tok/s; cold prefill 185 s (about 1,230 tok/s) |
| prefill | 1.75 s at 4K; 15.2 s at 30K (about 1,990 tok/s) |
| copy drafts: re-quote / code edit / chat | **939 / 880** / 49.6 tok/s against 49.6 / 50.4 / 48.7 plain |
| chats at once (`--parallel 4 --context 65536`) | 49.5 / 94.2 / 182.4 tok/s aggregate at 1 / 2 / 4 chats |
| shutdown (SIGINT) | exit 0 in 1.0 s, no process left; the 132-byte `SafeAllocator` line as on the Spark |

Fidelity of the packs against BF16 (teacher-forced on the Spark, the same `rtx` binary): [CUDA](CUDA.md#fidelity).
`st-6.0` is 0.99979 / 236 of 236 top-1, closer to BF16 than MLX q8.

### Host-RAM prompt cache

The profile keeps finished prompt states in 8 GiB of pinned host RAM (0 device bytes; the engine peak stays 12.49 GiB).
The same prompt twice:

| prompt | cold first token | restored | replies |
| ---: | ---: | ---: | --- |
| 30,215 tokens | 15.3 s | 0.85 s | identical |
| 122,601 | 78.0 s | **3.23 s** | identical |
| 227,182 | 185.0 s | 2.44 s | identical |

Restore is PCIe-bound, about 0.7-1.0 GiB/s through WSL2.

### NVMe tier

`--prompt-cache-gib 3 --spill-gib 8 --prompt-cache-disk DIR`: three cold prompts through a 3 GiB RAM budget, the same
three again, then a restart and the same three again.

| prompt | cold | again, same serve | after a restart | replies |
| ---: | ---: | --- | --- | --- |
| 29,505 | 12.8 s | **0.42 s** from disk | 0.39 s | identical |
| 98,217 | 50.1 s | **2.48 s** from disk (1,544 MiB in 1.6 s) | 2.47 s | identical |
| 116,565 | 61.7 s | 59.2 s: a miss ([KNOWN-ISSUES](KNOWN-ISSUES.md#rtx)) | **1.31 s** from disk | identical |

A state larger than the RAM budget never reaches the disk: at `--prompt-cache-gib 1` the 98K and 116K states were
refused (`1544 MiB passes the 1024 MiB budget`). Size the RAM cache to the longest prefix you want kept, about
1.6 GiB per 100K tokens (3.4 GiB for 227K).

## Build

On Linux or WSL2 with the CUDA 13 toolkit and Zig 0.17.0:

```bash
XGRAMMAR=xgrammar scripts/build-engine.sh rtx engine-rtx    # SM=120 by default (RTX 50); set SM= for another card
```

Without root, point it at a user-level toolkit with `NVCC=$CUDA_DIR/bin/nvcc`. The build took 6 minutes on the test
machine. For the WSL2 setup (driver, toolkit without sudo, Zig), follow the
[SenseNova recipe's WSL2 steps](https://github.com/jayleaton/sensenova-loop9b-tensorfold/blob/main/docs/RTX.md#windows-wsl2-steps);
they are the same for this model.

## Quantize

exllamav3 traces stock `qwen3_5` itself, so it calibrates in its own forward (250 x 2048 rows of its default set);
the SenseNova chain's transformers Hessians are not needed. `scripts/quantize-exl3.sh` clones exllamav3 at the
SenseNova packs' commit (`8c593e4e5c88`), makes a venv with torch for CUDA 13, and runs the engine's shim, `convert.py`
with the engine's `6.0` recipe flags (`-b 6.0 -hb 8 -eb 16 -cb mul1 --out_scales always`) and the attach step:

```bash
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir Qwen3.5-9B
NVCC=$CUDA_DIR/bin/nvcc scripts/quantize-exl3.sh engine-rtx Qwen3.5-9B st-6.0 6.0
```

On the 5070 Ti it took 36-77 s a layer and 166 s for `lm_head`, about half an hour. The pack leaves out `mtp.*` and the
vision tower, and records `"calibration": {"method": "exllamav3-native"}`. The test pack's `model.safetensors` sha256
is `7497f67bb53103c998de033d6d6beec4ee84b8f121931383718ecbfb823b2a0a` (8,255,878,403 bytes).

- Keep the BF16 checkpoint and the pack on ext4 under your home directory, not under `/mnt/c`: hashing and the
  checkpoint reader both fail on drvfs.
- The script does not pass `--no_swap`: on a 16 GB card `lm_head` runs out of memory without swapping weights to host.
- A stopped run resumes from `OUT.work`. exllamav3 then records the resumed run's `--out_scales` as `auto` in the
  pack's `config.json` and `quantization_config.json`; the tensors still carry their output scales.
- `4.0` and `5.0` work the same way. Their fidelity is in [CUDA](CUDA.md#fidelity); they were not run on the card.

## Serve

```bash
scripts/serve.sh rtx engine-rtx st-6.0    # 1 chat at 262,144 tokens, fp8 KV, 8 GiB host cache, NVMe tier
```

| profile | flags (`config/rtx/serve.args`) |
| --- | --- |
| default | `--parallel 1 --kv-cache fp8 --context 262144 --thinking-budget 2048 --prompt-cache-gib 8 --spill-gib 64` |

- **4 chats:** add `--parallel 4 --context 65536` (peak 12.93 GiB).
- `--prompt-cache-gib 8` holds one near-full prefix plus a few 32K ones. The engine caps it at half the RAM WSL sees,
  so the same line runs on a 32 GB host; give WSL more in `.wslconfig` for a larger cache.
- `--spill-gib 64` without `--prompt-cache-disk` uses the engine's default store under the home directory; keep it on
  ext4.
- If the desktop needs more of the card, cap the engine with `TENSORFOLD_CUDA_MEMORY_LIMIT_GB=14.5`.
- WSL's `nvidia-smi` can print a used figure near 2^44 MiB once the card is nearly full. Read the card from Windows
  (Task Manager, or `nvidia-smi` in a Windows terminal) and the engine's own peak from `/metrics` with
  `--metrics-open`.
