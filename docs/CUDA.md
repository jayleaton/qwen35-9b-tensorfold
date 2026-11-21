# DGX Spark (CUDA)

Measured on DGX Spark-class GB10 machines (128 GB unified memory, Ubuntu 24.04, driver 580, CUDA 13.0, Zig 0.17.0) on
2026-11-20 and 2026-11-21, with `config/cuda/`: upstream TensorFold `038e4a278` + `patches/cuda/` `0001`-`0032`, tree
`20a3fb077816`. Sections before [MTP drafting](#mtp-drafting) were measured with `0001`-`0027` (tree `9bac6ecbdef1`),
copy drafts only; `0028`-`0032` add MTP drafting and leave those numbers unchanged when it is off.

## Build

```bash
scripts/build-engine.sh cuda engine-cuda   # NVCC defaults to /usr/local/cuda/bin/nvcc
```

The build also writes `libtfgrammar` beside the server, from xgrammar v0.2.8 (commit `97787376faee`), which it clones
into `engine-cuda/.xgrammar`; `tool_choice: "required"`, a named function and `response_format` need it.
`XGRAMMAR=DIR` uses an existing v0.2.8 checkout instead, and `XGRAMMAR=none` skips the library (those requests then
get HTTP 400). On a machine without GitHub access, pass a TensorFold checkout or bundle holding the pin as the third
argument, and `XGRAMMAR=DIR`.

The engine and `tf-qwen35-cuda` built in about 8 minutes on one Spark; `libtfgrammar` adds a few.
`tensorfold-native --version` prints 1.2.9; your binary's sha256 will differ from ours with another build path or
toolchain.

## Serve

```bash
scripts/fetch-engine.sh metal engine-metal && scripts/convert.sh engine-metal Qwen3.5-9B q9-q4 4   # see CONVERT
scripts/mtp-head.sh Qwen3.5-9B q9-q4                    # the MTP head beside the tree (MTP drafting)
scripts/serve.sh cuda engine-cuda q9-q4                 # 8 chats, 262,144-token window, port 8090
PROFILE=batch scripts/serve.sh cuda engine-cuda q9-q4   # 16 chats at a 65,536-token window, for batch work
```

| profile | flags (`config/cuda/serve*.args`) |
| --- | --- |
| default | `--parallel 8 --context 262144 --kv-cache fp8 --max-tokens 32768 --thinking-budget 2048 --decode-share 0.5 --prompt-cache-gib 24 --spill-gib 256 --learn-daily-gib 256 --checkpoint-slots 8` |
| `batch` | the same with `--parallel 16 --context 65536` |

Both export `TENSORFOLD_MEMORY_RESERVE_GIB=36` (room for the 24 GiB host prompt cache in unified memory) and
`TF_QWEN_KV_PAGES=256`. The server is ready in 6-7 s, or 20 s with MTP drafting (the head is quantized at load). Its banner:

```
model 5.27 GiB; 8 streams at once, the engine's own at a 262144-token window (4.23 GiB, in the model's), 7 more
taking 7546 pages of 256 rows (29.94 GiB) as they need them ... of 75.3 GiB left after a 36.0 GiB reserve; ...
a 24.0 GiB prompt cache in host memory
serving q9 at http://127.0.0.1:8090/v1 (sampling: temperature 1.0, top_p 0.95; drafts: copies only; context: 262144)
```

## Results

`q9-q4`, the default profile unless noted; greedy, 256 tokens with `ignore_eos`, thinking off for the speed rows.
Prompts above 1K tokens are Python standard-library source. Every drafted reply was compared with its
`"draft": false` reply by sha256.

| check | result |
| --- | --- |
| `/v1/models`, chat, streamed reply | pass (`[DONE]` received) |
| tool call, `tool_choice` `auto` and `required` | pass: `get_weather({"city":"Paris"})`, `finish_reason: tool_calls` |
| decode, 21-token prompt | **46.4 tok/s** drafted and plain; replies identical |
| decode after a 30,209-token prompt | 41.5-41.9 drafted, 41.1 plain; identical; cold prefill 9.1 s (about 3,300 tok/s) |
| chats at once, aggregate | 1: 45.6; 4: 162.5; 8: 290.9; 16 (`batch`): 478.8 tok/s |
| copy drafts, re-quote a 6,000-character passage | **1,447 tok/s** against 45.7 plain; identical |
| copy drafts, rename a variable through the same file | **865 tok/s** against 45.7 plain; identical |
| copy drafts, free prose | 45.9 against 45.9 (nothing to copy) |
| prompt cache, 95,109-token prompt | cold first token 34.6 s; second turn from host RAM **0.54 s** |
| NVMe tier, the same prompt after a restart | **1.87 s**: 94,208 tokens read from disk (1,544 MiB in 1.26 s); reply identical to the first boot's |
| memory | host MemAvailable fell by 45 GiB at peak (default profile, 36 GiB reserve included), 25 GiB with `batch` |
| shutdown (SIGINT) | drains in 1-5 s, exit 0, no process left |
| soak, 30 minutes, 4 chats of mixed requests | **857 requests, 0 failures** (chat, stream, tool calls `auto` and `required`, coding with thinking, 8K-token summaries); RSS 0.77 GiB after warm-up, 0.80 at the end; MemAvailable flat (77.8 to 77.6 GiB) |

`q9-q8` (8-bit, 9.44 GiB): 25.2 tok/s at 21 tokens, 23.5 after 30,209, about 87 aggregate at 4 chats; drafted and
plain identical.

## Fidelity

Teacher-forced logits from `tf-qwen35-cuda MODEL step IDS OUT` on three fixed prompts (prose, code, reasoning) with
64-token continuations, 236 positions, each tree against the BF16 checkpoint on the same binary:

| tree | mean cosine | worst cosine | top-1 | mean top-5 overlap |
| --- | ---: | ---: | ---: | ---: |
| `q9-q8` | 0.99983 | 0.9955 | 235/236 | 4.90 |
| `q9-q4` | 0.98484 | 0.8647 | 222/236 | 4.39 |
| EXL3 6.0 bpw (`rtx` build) | 0.99979 | 0.9985 | 236/236 | 4.91 |
| EXL3 5.0 bpw | 0.99929 | 0.9966 | 235/236 | 4.85 |
| EXL3 4.0 bpw | 0.99762 | 0.9843 | 231/236 | 4.74 |

The `cuda` and `rtx` binaries give bit-identical BF16 and q4 logits, and the `metal` converter's `q9-q4` gives
bit-identical logits to the `cuda` converter's tree (f32 gated norm, MTP kept) on all three prompts. q4 loses the most
per token, yet with thinking on it scores as well as q8 on the benchmarks below and in the study behind them (q4 to q8:
+0.5 BFCL, +1.8 HumanEval+).

## Quality

The serving study's own kit, greedy (`temperature: 0`), `--thinking-reserve 2112` added to the serve line, 16 requests
at once. References are stock Qwen3.5-9B q4 in the study (same engine tree, its own server flags):

| check | this recipe (`batch` profile) | study reference |
| --- | ---: | ---: |
| HumanEval+ 164, thinking budget 1024 | **146-147/164 = 89.0-89.6** | 147/164 = 89.6 |
| BFCL v4 held-out 450, thinking budget 512 | **377/450 = 83.8** | 371/450 = 82.4 |

With the study's own server flags (`--context 33024 --parallel 16 --prompt-cache-gib 0`, bf16 KV) this build and tree
reproduce the study's 164 HumanEval+ replies token for token. The recipe profile's fp8 KV cache changes tokens (9 of
164 replies are identical) without moving the score. The same 164 replies scored 146 once and 147 once: one
HumanEval+ test is timing-dependent. Upstream main gave the same 614 replies as `cuda`, token for token.

## Upstream main

`config/upstream/` (upstream `main` `727153cb3`, no patches) on the same Spark and tree, same flags:

| check | upstream main | `cuda` |
| --- | --- | --- |
| API rows, tool calls `auto` and `required` | pass | pass |
| decode at 21 / 30,209 tokens | 46.1 / 41.2 tok/s, same replies as `cuda` | 46.4 / 41.1 |
| copy drafts, re-quote / code edit | 1,441 / 871 tok/s | 1,447 / 865 |
| 4 chats / 16 chats aggregate | 162.4 / 452.7 tok/s | 162.5 / 478.8 |
| host-RAM prompt cache, 95K prompt | 36.1 s cold, 0.58 s warm | 34.6 / 0.54 s |
| NVMe tier after a restart | **not restored**: 36.2 s, `imprint: saving 94208 tokens skipped (StagingFull)` | 1.87 s (`cuda` 0016) |
| stock EXL3 pack | **refused**: f32 `linear_attn.norm` (`UnexpectedTensor`) | the `rtx` build serves it |
| the `cuda` tree converter's `q9-q4` (f32 gated norm) | **refused** | serves |

Upstream main serves this model on a Spark with the converted tree, at the same speed and with the same greedy
replies. What it lacks for this recipe is the long-prompt disk tier and the EXL3 packs.

## MTP drafting

On by default in the default profile. The checkpoint carries one MTP layer (15 `mtp.*` tensors, BF16); the `cuda`
build drafts with it through upstream PR [#3582](https://github.com/ashhart/TensorFold/pull/3582)'s dense MTP drafter,
backported to the pin (`cuda` `0028`-`0029`). The head sits beside the tree as `mtp.safetensors`
(`scripts/mtp-head.sh`, [CONVERT](CONVERT.md#the-mtp-head)) and is quantized to affine-4 tables at load (`0030`).
A stream with a copy ahead takes the copy without an MTP pass (`0031`), and a shared round prices the MTP hold as a
fixed cost (`0032`, `TF_MTP_FIXED_HOLD=1` in `serve.env`; below). `scripts/serve.sh` adds `--drafter TREE` when
the tree holds the head and the profile has a `.mtp` file (`config/cuda/serve.mtp`). `MTP=0` serves copy drafts
only; `PROFILE=batch` stays copies only, because 16 short chats at once are where MTP does not pay (below; `MTP=1`
turns it on).

```
qwen mtp: the checkpoint's bf16 MTP layer drafts as affine-4 tables quantized at load (drafts only)
serving q9 at http://127.0.0.1:8090/v1 (sampling: temperature 1.0, top_p 0.95; drafts: MTP (q9-q4) + copies; ...)
```

The recipe build (`0001`-`0032`) on one GB10, default profile, greedy, 256 tokens with `ignore_eos` and thinking off.
Every row's replies were compared, stream by stream, with the same prompts sent with `"draft": false`. Columns:
- **MTP + copies:** the default;
- **MTP, no `0032`:** `0001`-`0031`, the same as `0032` with `TF_MTP_FIXED_HOLD` unset;
- **copies only:** the same tree without the head;
- **plain:** `"draft": false`;
- **previous recipe:** `0001`-`0027`, copy drafts.

Chat rows show per-stream decode / aggregate tok/s. The aggregate counts the whole wave, including the other
streams' prefill: prompts at 4K and 28K are not served from the prompt cache in this cell.

| cell | MTP + copies | MTP, no `0032` | copies only | plain | previous recipe | replies |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 chat, 52-token prompt | **78.2 / 77** | 73.5 / 73 | 50.2 / 50 | 47.0 / 47 | 50.4 / 50 | identical |
| 4 chats, 52 tokens | **62.5 / 226** | 62.3 / 229 | 45.4 / 168 | 42.5 / 167 | 44.8 / 168 | identical |
| 8 chats, 52 tokens, first wave | **49.0 / 358** | 49.6 / 351 | 38.4 / 298 | 38.4 / 299 | 38.5 / 299 | identical |
| 8 chats, 52 tokens, waves 2-6 | **48.4-50.2 / 349-359** | 36.7-37.1 / 255-266 | | 38.5 / 298 | | identical |
| 1 chat, 4,057-token prompt | **79.2 / 61** | 75.4 / 59 | 46.5 / 40 | 46.2 / 40 | 46.7 / 40 | identical |
| 4 chats, 4,057 tokens | **53.4 / 92** | 55.1 / 92 | 36.5 / 78 | 36.4 / 77 | 36.6 / 78 | identical |
| 8 chats, 4,057 tokens | **46.7 / 100** | 47.3 / 99 | 31.8 / 92 | 31.5 / 91 | 31.6 / 91 | identical |
| 1 chat, 28,322-token prompt | **72.9 / 56** | 66.5 / 52 | 42.7 / 36 | 42.5 / 36 | 42.4 / 36 | identical |
| 4 chats, 28,322 tokens | **46.7 / 72** | 42.9 / 72 | 29.2 / 63 | 29.2 / 61 | 29.4 / 62 | identical |
| 8 chats, 28,322 tokens | **37.4 / 79** | 34.2 / 78 | 23.6 / 71 | 23.6 / 71 | 23.7 / 71 | identical |
| re-quote a 6,000-character passage (512 tokens) | **1,668** | 1,674 | 1,517 | 47 | 1,519 | identical |
| rename a variable through the same file | **1,177** | 1,171 | 908 | 47 | 906 | identical |
| free prose | **80** | 81 | 47 | 47 | 47 | identical |
| server ready | 21 s | 20 s | 8 s | | 7 s | |

- **HumanEval+ 164** (budget 1024, `max_tokens` 1536, 16 at once, `PROFILE=batch` with `--drafter` added): 145/164 =
  88.4 in 308 s with MTP and 145/164 in 333 s with copy drafts only; all 164 replies token-identical. The MTP head's
  drafts were accepted 70% of the time (112,123 of 160,038). 13 replies stop at the 1,536-token cap; each is a prefix
  of the reply in this page's earlier HumanEval+ runs, which ran past it. The shipped engine and this build both count
  thinking inside `max_tokens` (checked on one request), and the shipped recipe (`0001`-`0027`, copy drafts) scores the
  same 145/164 under this cap, with all 164 replies identical. The gap to 89.0-89.6 lies outside drafting.
- **30-minute soak**, default profile with MTP, 8 at once (tool calls, streaming, chat, long prompts, code): 1,085
  requests, 0 failures. Server RSS 0.44 to 0.86 GiB; host MemAvailable 68.2 GiB at the start, 67.2 at its lowest,
  68.0 at the end.
- **Two more GB10s agree.** They ran #3582's head and the backport without `0031`: 74-75 tok/s for 1 chat at 52 to
  28K tokens against 41-46 plain, and 8 chats at 28K at 71 against 64-65 aggregate.
  - Without `0031`, re-quotes ran 1,264-1,356 against copies only's 1,430-1,445, 5-13% slower.
  - With `0031` they run 8-10% faster than copies only, and code edits gain 12% over MTP without it.
- **Memory:** host MemAvailable fell by 53 GiB at peak, against 43-44 GiB with copy drafts. On unified memory this covers
  the drafter's ring and scratch at 8 streams and a 262,144-token window, and the 36 GiB reserve.
- **Fixed hold (`0032`).** Lanes fit a drafted shared round's time over the plain point as a per-row cost. An MTP
  hold costs about the same however few drafts it holds (a depth-1 hold for half of 8 streams took a round from 25 to
  43 ms), so the fit's slope rose as drafts fell, the allocator cut drafts again, and 8 short chats settled at 255-266
  tok/s after the first wave. With `TF_MTP_FIXED_HOLD=1`, greedy drafting rounds' extra rows and time feed a decayed
  least-squares fit: its intercept is the hold, a fixed cost of any drafting round, and its slope what a row costs at
  this context (until the rounds' rows spread enough to tell them apart, the per-row price stands). A greedy round
  then drafts deep or not at all. Same binary, the variable off and on, interleaved on one GB10:
  - 8 chats, 52 tokens, 12 waves each: 342-358 against 252-357 aggregate (plain 297).
  - 4 chats at 4K: -4.7% per stream against the variable off, outside run-to-run noise of about ±3% (two starts of
    the arm without it: 54.8-56.0 against 57.8-59.5 per stream); -1.8% aggregate, within it. Two earlier pairs, with a
    build that differed only in sampled rounds, gave -2.3% and +1.1% (18 waves each). Against copy drafts alone at 4K
    it is +46% per stream (53.4 against 36.5).
  - 4 chats at 28K: -1.3% per stream, -0.2% aggregate.
  - 1 chat: +2.8% at 4K, -3.8% at 28K. One chat never reaches the change (shared prices start at two streams); the
    arm without it alone moved 66-75 tok/s between two starts at 28K.
  - Sampled traffic (the server's default temperature 1.0) keeps the per-row price. Applied to sampled rounds too, the
    fit drafted half as much on the soak's mixed requests (8 at once, tool calls, thinking, long prompts) and served
    7-13% fewer in 10 minutes. With greedy rounds only, the same soak served 368 requests against 344 with the
    variable off (0 failures), in line with `0031`'s 1,085 in 30 minutes.
  - HumanEval+ in `PROFILE=batch` with MTP and fixed hold: the same 164 replies in 304 s.
- **16 short chats at once** (`PROFILE=batch` with `MTP=1`): 468-476 against 493-494 aggregate with copy drafts alone
  (plain 490), so the batch profile keeps copy drafts only. With 4K prompts 16 chats draft ahead, 107 against 102.
- **Pin:** #3582 rebased onto upstream main `4c616e63d` (its head `5b2be93de`) ran on the same GB10 within 1% of this
  pin in every plain and MTP cell, with the same replies. The recipe keeps pin `038e4a278` with the backport.
