Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# Qwen3.5-9B on TensorFold: DGX Spark, RTX, Macs and Ally X

A recipe for serving stock [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) with
[TensorFold](https://github.com/ashhart/TensorFold) on a DGX Spark, a 16 GB RTX card and Apple Silicon, with the ROG Ally X (HIP) to follow. It ships exact
upstream TensorFold pins with the patch series of the
[SenseNova-Loop-9B recipe](https://github.com/jayleaton/sensenova-loop9b-tensorfold) (copy drafts in shared rounds,
the host-RAM and NVMe prompt-cache tiers, EXL3 packs on RTX), a one-step conversion from the official BF16 checkpoint,
serving profiles with the thinking budgets a serving study picked, and the measurements behind them.

## Credits

- **Model: [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) by the [Qwen team](https://huggingface.co/Qwen)**,
  Apache License 2.0. Download it under the terms on its model card.
- **Engine: [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart (ashhart) and the TensorFold
  contributors**, Apache License 2.0 (code written before 0.6.0 keeps its MIT notice). The `qwen3_5` CUDA family and
  the dense `qwen27` Metal family, their Gated DeltaNet and hybrid-attention kernels, the MLX affine and EXL3 loaders,
  copy drafts and the prompt cache are upstream work. The patches here extend it; they are not upstream changes.
- **Shared-round drafter pricing:** upstream TensorFold PR [#2475](https://github.com/ashhart/TensorFold/pull/2475)
  by [jrb-ai](https://github.com/jrb-ai).
- **EXL3:** [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp, built on
  [QTIP](https://github.com/Cornell-RelaxML/qtip) (Cornell RelaxML).

More in [CREDITS](docs/CREDITS.md); license terms in [LICENSING](docs/LICENSING.md) and [NOTICE](NOTICE).

## Pick your device

Each row was built and run end to end, except the Ally X's, which waits on its device run. Every build checks the exact
engine tree before it compiles (`config/<config>/tensorfold.tree`). Numbers are one chat unless marked.

| device | weights | engine (`config/`) | speed | quality | limits |
| --- | --- | --- | --- | --- | --- |
| **DGX Spark** (GB10, 128 GB) | 4-bit MLX g64 (`q9-q4`) | `cuda`: pin `038e4a278` + `patches/cuda` `0001`-`0027` | decode 46.4 tok/s, 41.1 after 30K; 8 chats 291, 16 chats 479 tok/s aggregate; copy drafts: re-quote 1,447, code edit 865 tok/s; a 95K prompt's next turn in 0.54 s (host RAM) or 1.87 s after a restart (NVMe) | HumanEval+ 89.0-89.6, BFCL held-out 83.8; top-1 222/236 against BF16; 30-minute soak, 857 requests, 0 failures | images not covered |
| **RTX 5070 Ti 16 GB** (WSL2) | EXL3 6.0 bpw (`st-6.0`), quantized here | `rtx`: pin `038e4a278` + `patches/cuda` `0001`-`0022` + `patches/rtx` `0001`-`0003` | decode 48-50 tok/s at 4K, 43-44 at 30K; 4 chats 182 tok/s aggregate; copy drafts: re-quote 939, code edit 880 tok/s; a 122K prompt's next turn in 3.2 s (host RAM), a 116K one in 1.3 s after a restart (NVMe) | top-1 236/236 against BF16 | the whole 262,144-token window at one chat, 12.49 GiB on the card beside the Windows desktop |
| **Mac Studio M5 Ultra** | `q9-q4` | `metal`: pin `6d0a128b0` + `patches/metal` `0001`-`0044` | decode 121.7 tok/s, 106.2 after 32K; prefill 4,400-4,950 tok/s; copy drafts: re-quote 757, code edit 238 tok/s | HumanEval+ 87.8, BFCL held-out 81.3 | 4 chats at 65,536 tokens; Macs with 48 GB or more |
| **16 GB Mac** | `q9-q4` | `metal`, `PROFILE=16gb` | as the Studio's per chip | | 1 chat at 81,920 tokens; peak 7.8 GiB (M5) to 9.4 GiB (M4 Pro) at 32K; measured under a 16 GB working-set cap |
| **MacBook Pro M4 Pro** (24 GB) | `q9-q4` | `metal`, `PROFILE=24gb` | decode 44.5 tok/s, 29.9 after 32K; prefill about 400 tok/s; re-quote 169, code edit 77 tok/s | HumanEval+ 89.6, BFCL held-out 150-row subset 80.0 | 4 chats at 49,152 tokens; with everyday apps open, memory pressure admits one chat at a time |
| **ROG Ally X** (Radeon 780M, `gfx1103`, HIP) | `q9-q4` | `hip`: upstream `813ff4c`, no patches | pending device run | pending device run | the engine's host suite and loader test pass on the CPU; the GPU waits on a reboot |

**DGX Spark** ([CUDA](docs/CUDA.md)), CUDA 13 toolkit and Zig 0.17.0:

```bash
XGRAMMAR=xgrammar scripts/build-engine.sh cuda engine-cuda   # + libtfgrammar from an xgrammar v0.2.8 checkout
scripts/fetch-engine.sh metal engine-metal                   # the converter only; nothing is built
scripts/convert.sh engine-metal Qwen3.5-9B q9-q4 4
scripts/serve.sh cuda engine-cuda q9-q4                      # 8 chats, 262,144 tokens, fp8 KV, port 8090
```

**RTX 16 GB** ([RTX](docs/RTX.md)), Linux or WSL2 with the CUDA 13 toolkit and Zig 0.17.0:

```bash
XGRAMMAR=xgrammar scripts/build-engine.sh rtx engine-rtx     # SM=120 (RTX 50) by default
scripts/quantize-exl3.sh engine-rtx Qwen3.5-9B st-6.0 6.0    # exllamav3 at its pinned commit, about half an hour
scripts/serve.sh rtx engine-rtx st-6.0                       # 1 chat, 262,144 tokens, fp8 KV, 8 GiB host cache
```

**Macs** ([METAL](docs/METAL.md)), Xcode's Metal toolchain and Zig 0.17.0:

```bash
scripts/build-engine.sh metal engine-metal
scripts/convert.sh engine-metal Qwen3.5-9B q9-q4 4
scripts/serve.sh metal engine-metal q9-q4                    # 48 GB+; PROFILE=24gb or PROFILE=16gb for smaller Macs
```

Every command above needs the checkpoint downloaded first ([Quick start](#quick-start)).

## Serving defaults

Thinking is on (the checkpoint's chat template). The serve profiles cap it at **2048** tokens, the agentic default:
in OpenCode workflows graded by running what the model built, stock Qwen3.5-9B scored the same from 2048 to 8192 and
2048 finished every turn, 30% faster than 4096. For single requests, send `"thinking_budget": 1024` for coding and
`512` for tool calls; with thinking on, HumanEval+ goes from 78.7 to 89.6 and BFCL held-out from 77.8 to 82.4.
The checkpoint has no `generation_config.json`, so `scripts/convert.sh` writes Qwen's temperature 1.0 / top_p 0.95
(without it the engine samples greedily). Copy drafts are on by default and exact. Details: [SERVING](docs/SERVING.md).

## Versions

| `config/` | upstream pin | patches | engine tree | verified | status |
| --- | --- | --- | --- | --- | --- |
| `cuda` | `038e4a278` | `cuda` `0001`-`0027` | `9bac6ecbdef1` | 2026-11-20 | DGX Spark |
| `rtx` | `038e4a278` | `cuda` `0001`-`0022`, `rtx` `0001`-`0003` | `64ba7d4085cc` | 2026-11-20 | 16 GB RTX |
| `metal` | `6d0a128b0` | `metal` `0001`-`0044` | `d33e2d93ea37` | 2026-11-20 | Macs |
| `hip` | `813ff4c` | none | `f70d190f1d1b` | CPU checks only | Ally X, pending device run ([HIP](docs/HIP.md)) |
| `upstream` | `727153cb3` | none | `c4e66e4220a8` | 2026-11-20 | control only ([CUDA](docs/CUDA.md#upstream-main)) |

The series are the SenseNova-Loop-9B recipe's, byte for byte, plus `rtx` `0003` (the EXL3 attach tool records how a pack
was calibrated); their loop and mHC code stays idle for a stock checkpoint. What the stock model takes from them, and
what upstream `main` lacks: [ENGINE](docs/ENGINE.md).

## Quick start

Needs `git`, Zig 0.17.0 (the engine's `.zig-version`), and Python 3.11+ with `numpy`, `torch` and `safetensors` for the
converter.

```bash
git clone https://github.com/jayleaton/qwen35-9b-tensorfold.git
cd qwen35-9b-tensorfold
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir Qwen3.5-9B
```

Then build and serve for your device ([Pick your device](#pick-your-device)), and:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"q9","messages":[{"role":"user","content":"What is 17 * 23?"}],"max_tokens":4096}'
```

Keep `max_tokens` above the thinking budget (2048), or the reply can end inside the reasoning. Send
`"chat_template_kwargs": {"enable_thinking": false}` to turn thinking off for one request. To check a patch series
without building, run `scripts/fetch-engine.sh cuda|rtx|metal|hip|upstream DIR`: it fetches the pin, applies the series
and checks the resulting tree.

## Not yet

- **MTP drafting (follow-up, measured):** the recipe ships copy drafts. The checkpoint's MTP head drafts on a Spark with
  an upstream draft PR's dense MTP drafter: 62.7 against 45.8 tok/s for one chat, 198 against 162 for four, replies
  token-identical; no released engine has it yet ([CUDA](docs/CUDA.md#follow-up-mtp-drafting)).
- **DFlash drafting:** TensorFold reads DFlash2 drafters only, and none is published for this model
  ([KNOWN-ISSUES](docs/KNOWN-ISSUES.md)).
- **Images.**
- **ROG Ally X numbers:** the engine and tree are checked on its CPU; the GPU runs wait on a reboot
  ([HIP](docs/HIP.md)).

## Documentation

| | |
| --- | --- |
| [MODEL](docs/MODEL.md) | the checkpoint: architecture, cache sizing, memory |
| [CONVERT](docs/CONVERT.md) | BF16 to MLX affine, and the sampling file |
| [SERVING](docs/SERVING.md) | thinking budgets, sampling, parallel requests, tool calls |
| [CUDA](docs/CUDA.md) | DGX Spark: build, serve, speed, cache tiers, fidelity, quality, upstream main |
| [RTX](docs/RTX.md) | EXL3 on a 16 GB RTX card |
| [METAL](docs/METAL.md) | Apple Silicon |
| [HIP](docs/HIP.md) | ROG Ally X, pending device run |
| [ENGINE](docs/ENGINE.md) | the pins and what the series give a stock checkpoint |
| [KNOWN-ISSUES](docs/KNOWN-ISSUES.md) | what is not solved |
| [CREDITS](docs/CREDITS.md) / [LICENSING](docs/LICENSING.md) | who built what, and under which terms |

## Licensing

This recipe's scripts, patches and docs: **Apache License 2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). The
TensorFold source the patches modify keeps TensorFold's license and notices, and the model keeps the Qwen team's
terms: [`docs/LICENSING.md`](docs/LICENSING.md).
