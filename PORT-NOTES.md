# MTP drafting: what RTX needs

2026-11-21. The CUDA (DGX Spark) build drafts with the checkpoint's own MTP head and ships it on by default
([CUDA](docs/CUDA.md#mtp-drafting)). The Macs (`metal` `0045`-`0050`, [METAL](docs/METAL.md#mtp-drafts)) and the Ally X
(the HIP pin as it stands, [HIP](docs/HIP.md#mtp-drafts)) draft with it too. This file lists what the RTX build needs;
its GPU run is pending.

## What the Spark build does

- **Head beside the tree.** `scripts/mtp-head.sh BF16_DIR TREE` writes `TREE/mtp.safetensors`. It holds the
  checkpoint's 15 `mtp.*` tensors, their BF16 bytes copied as shipped (486,582,880 bytes), using Python's standard
  library only. The file's sha256 is `1ea4b4ec525e67cd7bc63de20dddca18248ef22ef2d3f2987eec6eb54a36d896` on any
  machine; a file with the same tensors but another header layout has another sha.
  - The tree's shards and `model.safetensors.index.json` are unchanged.
  - An engine without MTP never opens the file, because shards are found through the index. On the Spark, the
    shipped `cuda` build serves the tree with the file beside it: same banner (`copies only`), same speeds.
- **Engine:** pin `038e4a278` with `cuda` `0001`-`0032`. The patches added for MTP:
  - `0028`: upstream PR [#3582](https://github.com/ashhart/TensorFold/pull/3582)'s prerequisite (`478c6c4f2`,
    since landed as #4132).
  - `0029`: #3582's dense MTP drafter (`b15403929`), backported.
  - `0030`: `cuda_mtp_weights.openHead` adds `mtp.safetensors` to the head's checkpoint when the index lists no
    `mtp.*` names.
  - `0031`: the MTP drafter declares `copy_first`, and lanes keep it beside context copies. A stream with a copy ahead
    takes the copy without an MTP pass, so re-quotes run faster than copy drafts alone instead of 13% slower.
  - `0032`: with `TF_MTP_FIXED_HOLD=1` (set in `config/cuda/serve.env`), a greedy shared round prices the MTP hold
    as a fixed cost plus a row slope, fit over drafting rounds. Before it, lanes fit the hold as a per-row cost,
    drafted less each round, and 8 short greedy chats settled 11-17% below plain decode after the first wave; with it
    they stay at 349-359 against plain's 297. Sampled rounds (the server's default temperature 1.0) keep the per-row
    price: on mixed sampled traffic the fit drafted half as much and served 7-13% fewer requests. Off by default.
- **Serving:** `scripts/serve.sh` adds `--drafter TREE` when the profile has a `.mtp` file and the tree holds
  `mtp.safetensors`. Only `config/cuda/serve.mtp` exists (the 8-chat default profile); `PROFILE=batch` stays copies
  only. `MTP=0` turns it off, and `MTP=1` asks for it on any profile. The loader quantizes the BF16 head to affine-4
  tables at load, and the banner reads `drafts: MTP (<tree>) + copies`.
- **Exactness gate:** every drafted reply equals its `"draft": false` reply, token for token (all cells, three GB10s).
- **A9 (the f32 `linear_attn.norm`) is not in the way.** The recipe's tree is written by the `metal` converter, which
  already stores that norm in bf16 (`metal` 0037).
- **Pin: backport, not bump.** On the same GB10, #3582 rebased on main `4c616e63d` (`5b2be93de`) gives the same plain
  decode as pin `038e4a278` within 1% in every cell (47.2 vs 47.3 tok/s for 1 chat, 298 vs 298 for 8 chats), and the
  same MTP speeds. The plain-decode gain an RTX card sees on the new base is specific to that device.
- **Batch profile stays copies only:** at 16 short chats, MTP with fixed hold runs 468-476 tok/s aggregate against
  493-494 for copy drafts alone; at 16 chats with 4K prompts it is 106-107 against 102.

## RTX (16 GB, EXL3 `st-6.0`)

1. **Engine:** `cuda` `0028`-`0032` apply cleanly on the `rtx` series (`cuda` `0001`-`0022` + `rtx` `0001`-`0003`);
   they become `rtx` `0004`-`0008`. Build for `SM=120`, then run the TensorFold tree's
   `python3 tools/pr_check.py --base <series tip>` and `zig build test-server-cpu`. `zig/dense_serving_test.zig`'s
   launch-sequence hash is the same on both series (`8085c4dc…`).
2. **Head beside the pack:** `scripts/quantize-exl3.sh` writes no `mtp.*`. Run `scripts/mtp-head.sh BF16_DIR st-6.0`;
   its sha must be the one above.
3. **Gate to check first:** #3582 refuses an EXL3 target only when it is MoE (`cuda_drafter_open.zig`), so a dense
   EXL3 target should open MTP. The MTP layer reads the target's embedding and output head, which are EXL3 tensors in
   this pack. Confirm that the banner shows MTP and that drafted replies equal `"draft": false`. If not, report the
   error; do not change the gate.
4. **Memory:** the head adds about 140 MB of affine-4 tables (4.5 bits a weight) plus the drafter's scratch and ring.
   The profile is one chat at 262,144 tokens with 12.49 GiB on the card. Re-measure the peak and the free margin
   beside the Windows desktop, and lower `--context` only if the measurement says so.
5. **Default:** RTX serves one chat, where MTP wins most on the Spark. Ship `config/rtx/serve.mtp` only after
   parity and the cells below hold.
6. **Pin:** the new base's plain-decode gain on RTX (about 65 against 48-50 tok/s at 4K) is a separate decision from
   MTP. Measure it the same way (plain, copies, MTP, parity) before proposing an `rtx` pin bump.
7. **Cells:** decode at 4K and 30K, re-quote and code edit, drafted against plain against copies only, parity on
   each; startup. Top-1 against BF16 is unchanged by drafting.

## HIP (ROG Ally X, `gfx1103`): done

The pin drafts with the head as it stands, no engine patch. Its `Checkpoint.mtp()` opens side files with the tree's
own quantization, so `scripts/mtp-sidecar.py` writes the head as a 4-bit g64 side file with the converter's quantizer
and norm fold. Decode 19.0 to 26.8 tok/s (+41%), 2 chats 33.7 to 39.8, every reply token-identical, 0.1 GiB more
memory: [HIP](docs/HIP.md#mtp-drafts).
