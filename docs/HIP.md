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

## MTP drafts

2026-11-21. **The pin drafts with the checkpoint's MTP head as it stands: no engine patch.** Its HIP `qwen3_5` path
(`backend/hip/engine/mtp.zig`, lanes in `hip_lanes.zig`) reads the head from any `mtp*.safetensors` beside an affine
tree. The converter drops `mtp.*`, so [`scripts/mtp-sidecar.py`](../scripts/mtp-sidecar.py) writes the head as a
side file with the converter's own 4-bit g64 quantizer and `1 + w` norm fold, from the same BF16 source. `q9-q4`
stays unchanged; `q9-q4-mtp` links to its files plus the side file:

```bash
mkdir q9-q4-mtp
python3 -I scripts/mtp-sidecar.py --source Qwen3.5-9B \
  --converter engine-metal/tools/sensenova_loop_mlx.py --out q9-q4-mtp/mtp.safetensors
for f in q9-q4/*; do ln -s "../q9-q4/${f##*/}" q9-q4-mtp/; done
tensorfold-native serve q9-q4-mtp --context 8192 --parallel 2 --prompt-cache-gib 0 --no-update-check \
  --mtp-drafts 2 --mtp-confidence 0.4
```

The device's side file: 15 source tensors written as 31, 136,884,552 bytes, sha256
`5676c42e20413cad34ebb12ba9f4dd0ad4dec886c6fc7474a4685361d3842b6e`. The banner reads `drafts: MTP + copies`, model
4.82 GiB (4.69 without the head). `--mtp-drafts` and `--mtp-confidence` are read by the HIP serve but hidden from
`serve --help`.

Same build and GTT limit as above, greedy, 256-token replies, prompt cache off unless noted, engine defaults (3 drafts,
confidence 0.3):

| measure | copies only (`q9-q4`) | MTP + copies (`q9-q4-mtp`) | MTP acceptance |
| --- | --- | --- | --- |
| decode, 21-token prompt, 1 chat | 19.0 tok/s | **26.8** (+41%) | 49-53% |
| decode, 7,257-token prompt, 1 chat | 17.1 | **24.5** (+43%) | 60-61% |
| 4,395-token prompt, 128-token thinking replies, 12 runs | 17.8 | **26.9-28.4** (+57%) | 61-67% |
| 2 chats aggregate, `--parallel 2` (3 runs) | 33.7 | **39.8** (+18%) | |
| re-quote (1,515-token prompt, 512 tokens) | 57.0 | **62.8** | 98% |
| code edit (1,525-token prompt, 512 tokens) | 53.3 | **54.7** | 94% |
| chat, 512 tokens | 18.5 | **25.6** | 51% |
| prompt cache, resumed against fresh (5K prompt, `--prompt-cache-gib 1`) | 13.4 s to 0.30 s | 13.4 s to 0.33 s | |
| startup | 13.0-16.4 s | 13.1-15.4 s | |
| VRAM after the rows, `--parallel 2`, cache off | 6.5 GiB | 6.6 GiB (6.8 at `--parallel 1`) | |

- **Parity:** every drafted reply is token-identical to `"draft": false` and to the copies-only server's reply. The
  engine's `zig/tests/qwen35/serve_check.py` passes its 14 exactness contracts with drafting live, at greedy and at
  temperature 0.8.
- **No hang:** every row ran with no MES or hang line in the kernel log; GTT use stays at 545 MiB with or without the
  head.
- Acceptance is below the Spark's 69.6%, so the gain is smaller. On copy-heavy replies MTP adds little: copies already
  accept 94-98%.

**Draft depth.** A sweep of depth 2-5 against confidence 0.2-0.5 (one `--parallel 2` server per cell; one chat at 21
and 4,395 tokens, and 2 chats) kept every reply token-identical across all cells. The whole surface sits within about
5%, the drift between identical cells on a handheld that warms up. Depths 4 and 5 (which need a depth patch not carried
here) never paid: acceptance falls to 51-56% on short prompts. **On the Ally X serve `--mtp-drafts 2 --mtp-confidence
0.4`.** In an interleaved pass against the engine default it was equal on short prompts (26.15 against 26.2 tok/s),
+2.6% at 4.4K (25.45 against 24.8) and +0.8% for 2 chats (37.65 against 37.35), with acceptance up to 58% short and
69-71% at 4.4K. The engine default stays 3 drafts at 0.3.
