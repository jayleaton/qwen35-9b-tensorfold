# MTP drafting: what RTX and HIP need

2026-11-21. The CUDA (DGX Spark) build drafts with the checkpoint's own MTP head and ships it on by default
([CUDA](docs/CUDA.md#mtp-drafting)). This file lists what the RTX and HIP builds need to do the same.

## What the Spark build does

- **Head beside the tree.** `scripts/mtp-head.sh BF16_DIR TREE` writes `TREE/mtp.safetensors`. It holds the
  checkpoint's 15 `mtp.*` tensors, their BF16 bytes copied as shipped (486,582,880 bytes), using Python's standard
  library only. The file's sha256 is `1ea4b4ec525e67cd7bc63de20dddca18248ef22ef2d3f2987eec6eb54a36d896` on any
  machine; a file with the same tensors but another header layout has another sha.
  - The tree's shards and `model.safetensors.index.json` are unchanged.
  - An engine without MTP never opens the file, because shards are found through the index. On the Spark, the
    shipped `cuda` build serves the tree with the file beside it: same banner (`copies only`), same speeds.
- **Engine:** pin `038e4a278` with `cuda` `0001`-`0031`. The patches added for MTP:
  - `0028`: upstream PR [#3582](https://github.com/ashhart/TensorFold/pull/3582)'s prerequisite (`478c6c4f2`,
    since landed as #4132).
  - `0029`: #3582's dense MTP drafter (`b15403929`), backported.
  - `0030`: `cuda_mtp_weights.openHead` adds `mtp.safetensors` to the head's checkpoint when the index lists no
    `mtp.*` names.
  - `0031`: the MTP drafter declares `copy_first`, and lanes keep it beside context copies. A stream with a copy ahead
    takes the copy without an MTP pass, so re-quotes run faster than copy drafts alone instead of 13% slower.
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
- **Known limit:** at 8 concurrent short chats, a server that has run a while drafts at 238-266 tok/s aggregate
  against plain's 289-298 (two GB10s). The first wave on a fresh server reaches 350. With 4K or 28K prompts, MTP stays
  ahead (+8-10% aggregate). The lanes rule's learned shared-round pricing is the cause ([KNOWN-ISSUES](docs/KNOWN-ISSUES.md)).

## RTX (16 GB, EXL3 `st-6.0`)

1. **Engine:** `cuda` `0028`-`0031` apply cleanly on the `rtx` series (`cuda` `0001`-`0022` + `rtx` `0001`-`0003`);
   they become `rtx` `0004`-`0007`. Build for `SM=120`, then run the TensorFold tree's
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

## HIP (ROG Ally X, `gfx1103`)

1. **Engine:** upstream `813ff4c` already drafts with a `qwen3_5` MTP head on HIP
   (`backend/hip/engine/mtp.zig`, `hip_lanes.zig`). No recipe patch is known to be needed.
2. **How it finds the head:** `weights/checkpoint.zig` `Checkpoint.mtp()` reads `mtp.*` from side files
   (`files.side`; its test writes `mtp.safetensors`) or from the main index. It opens them with the tree's own
   quantization (`Table.open(..., c.config.quant)`), so it may refuse the BF16 sidecar in an affine-4 tree.
   - First try `q9-q4` with the BF16 `mtp.safetensors` beside it.
   - If it is refused, try a 4-bit affine g64 copy of the head (the converter's MLX affine path, with norms handled as
     in the main tree), and record which one loads.
3. **Memory:** the Ally X shares system RAM with the GPU. Measure the drafter's extra bytes at the profile's window.
4. **Cells:** as for RTX, plus the HIP host suite on the pin.
