# Metal: Mac Studio, 16 GB Macs and the MacBook Pro

| | |
| --- | --- |
| Engine | upstream TensorFold `6d0a128b0` + [`patches/metal/`](../patches/metal/) `0001`-`0049`, tree `767bd3b7d1846ff4973616e7d4287392fbb9e000` ([`config/metal/`](../config/metal/)), checked 2026-11-21 (`0001`-`0044`, tree `d33e2d93`, on 2026-11-20) |
| Studio | Mac Studio, M5 Ultra, 64-core GPU, 96 GiB; macOS 27.2, Xcode 27.2, Zig 0.17.0 |
| MacBook Pro | M4 Pro, 20-core GPU (no tensor units), 24 GiB; macOS 26.3, Zig 0.17.0; Docker, Chrome and other apps left open, swap near full |
| 16 GB Macs | the Studio with Metal's working set capped to a 16 GB Mac's (`TF_QWEN27_WORKING_SET_BYTES=11453251584`, 10.67 GiB), checked on the M4 Pro under the same cap; no physical 16 GB Mac |
| Weights | `q9-q4` from [CONVERT](CONVERT.md): its four shards are byte-identical on the Studio, the MacBook and the DGX Spark |

The `metal` series `0001`-`0044` serves the stock checkpoint unchanged; `0045`-`0049` add a reader and drafter for its MTP
head ([MTP drafts](#mtp-drafts)). `tensorfold-native --version` prints 1.2.9.

## Build

On the Mac, with Xcode's macOS SDK and Metal toolchain and Zig 0.17.0:

```bash
scripts/build-engine.sh metal engine-metal          # 150 s on the Studio, 291 s on the MacBook
git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar.git xgrammar
(cd engine-metal && zig build grammar -Dcpu=apple_m1 -Dxgrammar="$PWD/../xgrammar")   # tool_choice required/named
```

The `apple_m1` target keeps the binary compatible with M1 through M5. Then convert on the Mac or anywhere else
([CONVERT](CONVERT.md)): `scripts/convert.sh engine-metal Qwen3.5-9B q9-q4 4`, 58 s on the Studio and 82 s on the
MacBook.

## Serve

```bash
scripts/serve.sh metal engine-metal q9-q4                  # 48 GB Macs and up: 4 chats at 65,536 tokens
PROFILE=24gb scripts/serve.sh metal engine-metal q9-q4     # 24 GB Macs: 4 chats at 49,152
PROFILE=16gb scripts/serve.sh metal engine-metal q9-q4     # 16 GB Macs: 1 chat at 81,920
```

With the MTP head (recommended on every profile; [MTP drafts](#mtp-drafts)), save it once from the BF16 checkpoint
and pass it to any of the lines above:

```bash
scripts/mtp-head.sh Qwen3.5-9B q9-mtp-head                 # config.json + mtp.safetensors (486,582,880 bytes)
scripts/serve.sh metal engine-metal q9-q4 --drafter q9-mtp-head
```

| profile | flags (`config/metal/serve*.args`) | for |
| --- | --- | --- |
| default | `--context 65536 --parallel 4 --thinking-budget 2048` | 48 GB or more; the Studio keeps 14.3 GiB wired |
| `24gb` | `--context 49152 --parallel 4 --thinking-budget 2048` | 24 GB Macs: the default is refused there (`--parallel 4 needs 10.8 GiB ... 10.7 GiB remains`); 8.5 GiB wired |
| `16gb` | `--context 81920 --parallel 1 --thinking-budget 2048` | 16 GB Macs: the largest window an M1-M4 admits under the 16 GB working set (90,112 is refused); an M5 admits up to 98,304 |

The first launch of a new build takes about 22 s while it times the draft windows; later launches 1.7-2.3 s. The
banner then reads:

```
info: qwen27: lone loop=lanes width=16; lane window=16, lane copies=on
[tensorfold] serving q9 at http://127.0.0.1:8090/v1 (sampling: temperature 1.0, top_p 0.95; drafts: copies only; context: 65536; loaded in 1.7s)
```

## Results

| | Studio M5 Ultra | 16 GB cap (Studio) | MacBook Pro M4 Pro 24 GB |
| --- | --- | --- | --- |
| profile | default | `16gb` window, memory rows | `24gb` |
| decode, one chat, 1K / 32K prompt | **121.7 / 106.2 tok/s** | 120.5 / 106.8 | **44.5 / 29.9** |
| cold prefill 4K / 8K / 32K | 4,949 / 4,892 / 4,444 tok/s | 4,760 / 4,768 / 4,370 | 406 / 397 / 345 |
| peak footprint, 4K / 8K / 32K window | 10.75 / 10.88 / 11.63 GiB | **6.94 / 7.07 / 7.82 GiB** | 8.48 / 8.61 / 9.36 GiB |
| copy drafts: re-quote / code edit | **757 / 238 tok/s** (6.2x / 1.95x) | | 169 / 77 (3.9x / 1.8x) |
| MTP drafts: 1K / 32K / re-quote / code edit ([MTP drafts](#mtp-drafts)) | **228 / 171 / 1,012 / 452 tok/s** | 32K chat 197 against 101 | **79 / 46 / 152 / 132** (before `0048`) |
| HumanEval+ 164, budget 1024 | **144/164 = 87.8** | same weights and engine | **147/164 = 89.6** |
| BFCL held-out, budget 512 | **366/450 = 81.3** | | 120/150 = 80.0 (the Studio: 116/150 on the same rows) |

Speed rows: greedy, 256 tokens with `ignore_eos`, thinking off, `--prompt-cache-gib 0`, prompts from the engine's
`docs/`. Each prompt was sent drafted and with `"draft": false`; all 8 cases gave identical tokens on both Macs (twice
on the MacBook), and identical prompts gave identical tokens on the M5 Ultra and the M4 Pro.

### Studio

- **API:** `/v1/models`, a thinking chat (2,049 reasoning tokens at the 2048 budget, then `391`), a streamed reply,
  tool calls `auto` and `required`: pass. SIGINT exits 0 in 0.19-0.66 s with requests drained.
- **The plan's probe** (`check.py`, prompt cache on): decode 123.0 drafted / 122.7 plain; 30,142-token prompt cold in
  6.77 s, then 108.4 / 108.4 tok/s; 4 chats 422 tok/s aggregate; copies with 1,500-token prompts and 512-token replies:
  re-quote 1,426 tok/s and code edit 1,144 tok/s against 122 plain. One reply sha per case.
- **4 chats at once, 1K each:** 317 tok/s aggregate with copy drafts, 329 without; copy drafts cost 3.6% there with
  little to copy.
- **`q9-q8`:** 77.2 / 70.8 tok/s at 1K / 32K, 18.5 GiB wired; HumanEval+ 146/164 = 89.0, BFCL held 380/450 = 84.4.

### 16 GB Macs

One fresh server per row, `--parallel 1 --prompt-cache-gib 0 --no-drafts`, one prompt filling the window less the
reply, peak from `/metrics`. Under the 10.67 GiB cap:

| window | Studio (M5) peak | M4 Pro peak | decode on the Studio |
| ---: | ---: | ---: | ---: |
| 4,096 | 6.94 GiB | 8.48 GiB | 120.5 tok/s |
| 8,192 | 7.07 | 8.61 | 117.3 |
| 32,768 | 7.82 | 9.36 | 106.8 |
| 65,536 | 9.33 | | 92.7 |
| 81,920 | | 9.61 | |
| 98,304 | 9.15 (128-row prompt step) | refused | 82.1 |

Admitted at startup under the cap: one chat up to 98,304 tokens on the M5 and 81,920 on the M4 Pro; two chats at
32,768 (M5) or 40,960 (M4 Pro). A chip without tensor units keeps attention partials for every key chunk, so it peaks
1.4-1.5 GiB higher; the `16gb` profile follows the smaller window. A 16 GB Mac that is also running other apps leaves
less than the cap assumes.

### MacBook Pro M4 Pro

The commands of [Build](#build) were typed as written, on a machine in everyday use. With swap near full the engine
sheds load under memory pressure: at `warning` it admits one stream at a time, and at `critical` it answers 503. Four
chats gave 52.7 tok/s aggregate when they ran concurrently and about 33 tok/s when admitted one at a time; one chat
decodes at about 45 tok/s.

## Quality

The serving study's kit, greedy, the serve profile plus `--thinking-reserve 2112`, 4 requests at once. Spark
references: stock q4 HumanEval+ 89.6, BFCL held 82.4.

The Studio fails 6 HumanEval+ problems that the Spark passes and passes 3 it fails: Metal and CUDA kernels differ, so
greedy tokens diverge on long replies (14 of 164 replies are identical across them). It is not a q4 effect: `q9-q8`
scores 89.0. The MacBook and the Studio differ by 3 problems for the same reason (simdgroup fragments against tensor
units).

Scoring on macOS needs Python 3.12 (`bfcl-eval==2026.3.23` does not install on 3.15) and evalplus's 4 GiB
`RLIMIT_AS` skipped on Darwin, which macOS refuses; with that edit in the scoring env, the Spark's own reference rows
rescore to 89.6.

## MTP drafts

`patches/metal` `0045`-`0047` read the checkpoint's MTP head (one gated attention layer and its MLP over the last
layer's state and the next token's embedding) and draft with it. `--drafter q9-mtp-head` loads its 15 `mtp.*` tensors
and quantizes the linears to affine 4-bit at load; the `q9-q4` shards are not touched. Each round the head drafts a
chain that stops after the draft taking the chain's probability under 0.5 (`TF_QWEN27_MTP_CONFIDENCE`) or at 8 drafts
(`TF_QWEN27_MTP_DEPTH`), and the target verifies it, so replies are the tokens of `"draft": false`. Copy drafts stay
on beside the head: a lone stream appends up to 48 copied tokens past the chain (`TF_Q27_COPY_ROWS`, 0 turns them
off), and several chats share copy rounds. `0048` (copy_first, the idea of `cuda` `0031` on the MTP recipe) starts
those copies at a 6-token match and, once a copy has landed whole, lets the next copy take the round without a chain;
a copy that misses hands the round back to the head. `0049` makes the serving line say `MTP (...) + copies` when the
lone loop copies and `MTP (...)` alone when it does not (`lane copies=off` names only the lane policy).

Studio, default profile, the speed rows above (drafted / `"draft": false`, tok/s; acceptance is accepted / drafted):

| case | MTP | acceptance | copies only |
| --- | --- | --- | --- |
| 1K / 4K / 8K prompt | **228 / 115**, 197 / 113, 202 / 110 | 55%, 46%, 55% | 126 / 122, 120 / 120, 116 / 115 |
| 32K prompt | **171 / 102** | 47% | 108 / 107 |
| re-quote / code edit | **1,012 / 115**, **452 / 116** | 63%, 64% | 763 / 121, 238 / 122 |
| story | 214 / 115 | 61% | 123 / 122 |
| 4 chats, 1K each (aggregate) | 303 / 305 | 69% | 316 / 352 |
| 4 chats, 32K each (each chat's decode) | **140-183 / 88-100** | 57% | 94-107 / 29-100 |
| startup | 11.6 s first launch of a build, 2.1 s after; 14.7 GiB wired (14.3 without) | | 11.6 s, 14.3 GiB |

- All 9 cases token-identical drafted and plain, on every run (11 servers, chain settings swept). HumanEval+ and BFCL
  held with the head: **144/164 = 87.8** and **366/450 = 81.3**, all 614 replies token-identical to the run without
  it; 76% and 71% of drafts accepted.
- The chain's cut was chosen from a sweep: a fixed 8 drafts accepted 26% and ran 1K at 174, 32K at 118; a fixed 3 was
  the best fixed length (1K 216, code edit 351); cuts of 0.3, 0.5 and 0.7 were within noise of each other, and
  capping them at 4 cost re-quote and code edit.
- 16 GB tier (Studio, 10.67 GiB cap, `16gb` profile with the head): admitted at 81,920 with 9.0 GiB wired; a 33K
  chat at 196.8 / 100.7 tok/s, same tokens, 9.64 GiB peak.
- MacBook Pro M4 Pro (`24gb`, everyday apps open): 1K **79.4 / 44.7**, 32K **45.6 / 30.6**, code edit **132.5 / 44.9**,
  story 63.4 / 45.0, re-quote 152.0 / 45.1 (copies alone: 186.5); acceptance 59-93%; every case token-identical.
  9.0 GiB wired with the head (8.5 without), 10.0 GiB footprint peak. HumanEval+ and BFCL (150-row subset) replies
  are identical to the run without it (164/164, 150/150), and the runs took 1,363 s and 680 s instead of 2,737 s and
  1,259 s.
- copy_first (`0048`): re-quote 686 -> 1,012 tok/s, above copies alone (763); code edit 441 -> 452; every other row
  within run-to-run noise (4K 196-201 over six servers). The 6-token match alone gave re-quote 1,002 and code edit
  438; copy_first on every round instead of after a landed copy dropped code edit to 329. Acceptance reads lower
  because the copies offer more tokens. The MacBook rows are from the build before `0048`; there, re-quote was 152
  with the head against 186 with copies alone.

## Upstream main on a Mac

Upstream `main` (`727153cb3`, `tensorfold-native` 1.3.0) serves `q9-q4` unpatched on the Studio, at the same plain
speed within 1% and with the same tokens on identical prompts. A lone stream never drafts copies there
(`lone loop=plain`, `lane copies=off`): the re-quote stays at 122 tok/s against 757 on the pin. That is `metal`
`0042`-`0043`. Its banner also calls the model "Qwen3.8-27B".

## Not covered on Metal

Images, the prompt-cache tiers and the soak (Spark and RTX only), and a
physical 16 GB Mac.
