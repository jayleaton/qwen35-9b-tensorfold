# MTP drafting: what RTX and HIP need

Draft, 2026-11-21. The CUDA (DGX Spark) port is in progress. This file says what the other devices need to draft with
the checkpoint's own MTP head. It is updated as the Spark work lands.

## What the Spark build does

- **Head:** the checkpoint's one MTP layer, its 15 `mtp.*` tensors in BF16 (486,582,880 bytes), saved as
  `mtp.safetensors` beside the converted tree by `scripts/mtp-head.sh BF16_DIR TREE`. The script uses Python's
  standard library only and copies the tensors' bytes as shipped. The result is sha256
  `1ea4b4ec525e67cd7bc63de20dddca18248ef22ef2d3f2987eec6eb54a36d896` on any machine; a file with the same tensors but
  another header layout has another sha. The tree's shards and `model.safetensors.index.json` stay unchanged, so a
  build without MTP reads exactly the tree it read before. An engine without MTP never opens the file, because shards
  are found through the index. The shipped `cuda` engine serving the tree with the file beside it is being checked on
  the Spark.
- **Engine:** the `cuda` series plus three patches:
  1. upstream PR [#3582](https://github.com/ashhart/TensorFold/pull/3582)'s prerequisite (`478c6c4f2`);
  2. its dense MTP drafter (`b15403929`), ported onto pin `038e4a278`;
  3. a recipe patch: `cuda_mtp_weights.openHead` adds `mtp.safetensors` to the head's checkpoint when the index lists
     no `mtp.*` names.
- **Drafter:** `--drafter <tree>`. The loader quantizes the BF16 head to affine-4 tables at load (`qwen mtp: the
  checkpoint's bf16 MTP layer drafts as affine-4 tables quantized at load`). Depth defaults to 3 (`--mtp-drafts`).
  The banner reads `drafts: MTP (<tree>) + copies`.
- **Exactness gate:** every drafted reply equals its `"draft": false` reply, token for token.
- **A9 (the f32 `linear_attn.norm`) is not in the way.** The recipe's tree is written by the `metal` converter, which
  already stores that norm in bf16 (`metal` 0037). Only the older `cuda`-converter tree, which the recipe does not
  ship, keeps it in f32.

## RTX (16 GB, EXL3 `st-6.0`)

1. **Engine:** the three MTP patches apply cleanly on the `rtx` series (`cuda` `0001`-`0022` + `rtx` `0001`-`0003`).
   Build them for `SM=120` and run the TensorFold tree's `pr_check --base` and `zig build test-server-cpu`.
   `zig/dense_serving_test.zig` pins a hash of the forward's launch sequence. The port updates it to `8085c4dc…`, and
   the `rtx` series gives the same hash.
2. **Head beside the pack:** `scripts/quantize-exl3.sh` writes no `mtp.*`. Put the same BF16 `mtp.safetensors` in
   the `st-6.0` directory with `scripts/mtp-head.sh`. It checks the tree's `model.safetensors.index.json`, so run it
   on the pack directory.
3. **Gate to check first:** #3582 refuses an EXL3 target only when it is MoE
   (`cuda_drafter_open.zig`: `if (exl3 and moe()) return error.Exl3DrafterUnsupported`), so a dense EXL3 target
   should open MTP. Nobody has run it yet. The MTP layer reads the target's embedding and output head, which are
   EXL3 tensors in this pack. Confirm that the MTP banner appears and that drafted replies equal `"draft": false`.
   If not, report the error; do not change the gate.
4. **Memory:** the head adds about 140 MB of affine-4 tables (4.5 bits a weight) plus the drafter's scratch and ring. The profile is
   one chat at 262,144 tokens with 12.49 GiB on the card. Re-measure the peak and the free margin beside the Windows
   desktop, and lower `--context` only if the measurement says so.
5. **Cells:** decode at 4K and 30K, re-quote and code edit, drafted against plain against copies only, parity on
   each; startup; top-1 against BF16 must be unchanged (drafting does not change the target).

## HIP (ROG Ally X, `gfx1103`)

1. **Engine:** upstream `813ff4c` already drafts with a `qwen3_5` MTP head on HIP
   (`backend/hip/engine/mtp.zig`, `hip_lanes.zig`). No recipe patch is known to be needed yet.
2. **How it finds the head:** `weights/checkpoint.zig` `Checkpoint.mtp()` reads `mtp.*` from side files
   (`files.side`; its test writes `mtp.safetensors`) or from the main index. It opens them with the tree's own
   quantization (`Table.open(..., c.config.quant)`), so the BF16 sidecar may be refused in an affine-4 tree.
   - First try `q9-q4` with the BF16 `mtp.safetensors` beside it.
   - If it is refused, try a 4-bit affine g64 copy of the head (the converter's MLX affine path, with norms handled as
     in the main tree), and record which one loads.
3. **Memory:** the Ally X shares system RAM with the GPU. Measure the drafter's extra bytes at the profile's window.
4. **Cells:** as for RTX, plus the HIP host suite (`zig build test-server-cpu`) on the pin.

## Open on the Spark side (this file follows them)

- **Pin bump or backport:** #3582 has been rebased to `5b2be93de`, a single commit on main `4c616e63d`; its
  prerequisite landed as #4132. Plain decode on that base is being measured against pin `038e4a278`. If the bump wins,
  the patch list above changes to the new pin plus a smaller series, and this file will say so.

- Speed table at 1/4/8 chats, 4K and 30K, re-quote and code edit: MTP + copies against copies only against plain.
- The default per workload, and whether re-quotes keep copy drafts' speed with MTP on.
- HumanEval+ parity, a 30-minute soak, startup.
