# Known issues

## All devices

- **MTP drafting everywhere but the RTX card.** The DGX Spark ([CUDA](CUDA.md#mtp-drafting)), the Macs, Studio and
  MacBook Pro ([METAL](METAL.md#mtp-drafts)), and the Ally X ([HIP](HIP.md#mtp-drafts)) draft with the checkpoint's
  MTP head. The RTX build (EXL3 pack) is pending its GPU run ([PORT-NOTES](../PORT-NOTES.md)). [`z-lab/Qwen3.5-9B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) is a DFlash v1
  drafter (block 16, one full-attention layer, no candidate selector); TensorFold's CUDA drafter reads DFlash2 only
  and refuses it with `BadDraftConfig`. No DFlash2 drafter for Qwen3.5-9B is published. Copy drafts are on and exact.
- **Images are not covered.** The converted tree carries the vision tower; the CUDA image path and Metal's images
  were not checked for this checkpoint.
- **Benchmark replies are not bit-stable across serving flags.** Greedy replies are identical across drafts on or off
  and across parallel widths, and the `cuda` build reproduces the serving study's replies under its flags; the fp8 KV
  cache in the recipe profiles changes some tokens (HumanEval+ 89.0-89.6 against 89.6, BFCL held 83.8 against 82.4).

## CUDA

- `tensorfold-native` prints one `error(SafeAllocator): leaked ... len: 132` line when it exits. Shutdown still
  drains and exits 0; nothing grows while serving.
- The `cuda` tree's own converter writes a tree only the patched CUDA loader reads. Convert with the `metal` engine's
  converter ([CONVERT](CONVERT.md)).
- Upstream `main` without the series serves the converted tree at the same speed, but its NVMe tier skips saving long
  prompt states (`StagingFull`) and it refuses the EXL3 packs ([CUDA](CUDA.md#upstream-main)).
- **MTP does not pay at 16 short chats at once**: 468-476 tok/s aggregate against 493-494 with copy drafts alone, so
  `PROFILE=batch` serves copy drafts only (`MTP=1` overrides). With 4K prompts, 16 chats draft ahead (106-107 against
  102) ([CUDA](CUDA.md#mtp-drafting)).
- **Startup takes 20 s with MTP** against 7 s with copy drafts only: the loader quantizes the BF16 head at load.

## RTX

- **A state can miss the NVMe tier in the same serve.** With a 3 GiB RAM budget, a 116K-token prompt's state was
  evicted from RAM by two restores and not read back from disk in that serve (59 s, a full prefill); after a restart
  the same state restored from disk in 1.3 s. States larger than `--prompt-cache-gib` never reach the disk.
- The RTX numbers were taken with the Windows desktop in use on the same card; one 4-chat cell in the mixed matrix
  (code edit, drafts off) is an outlier and was not re-run.
- Only `st-6.0` was run on the card. `4.0` and `5.0` were measured for fidelity on a Spark only.

## Metal

- **24 GB Macs refuse the default profile** (`--parallel 4 needs 10.8 GiB ... 10.7 GiB remains`); use `PROFILE=24gb`.
- **Memory pressure serializes chats.** On a 24 GB MacBook with everyday apps open and swap near full, the engine
  admits one stream at a time at `warning` pressure and answers 503 at `critical`.
- **No physical 16 GB Mac was tested.** The 16 GB rows are the Studio and the M4 Pro under a 16 GB working-set cap.
- **Metal and CUDA greedy replies differ on long outputs** (different kernels): HumanEval+ 87.8 on the Studio and 89.6
  on the M4 Pro against 89.6 on the Spark, with the same weights.

## HIP

- **GTT above the 8 GiB carve-out hangs the GPU in prefill.** With `ttm.pages_limit` at `4194304` (15.3 GiB) the engine
  budgets from host memory, places its buffers in GTT, and `gfx1103` stops answering (`MES failed to respond to
  msg=REMOVE_QUEUE`, three hangs in five minutes). At `1966080` (7.5 GiB, below the carve-out) every buffer lives in
  VRAM and 38 requests ran clean. Keep GTT below the carve-out on this device.
- **The GPU numbers come from a superset build, not the bare pin.** They were taken on `813ff4c` plus
  SenseNova-Loop's looped family, which stays idle for a stock checkpoint; the unpatched pin passed the CPU checks only.
- **Small windows only.** Speed was measured at one chat and 8,192 tokens and the quality rows at four chats and 4,096;
  larger windows were not run on the device.
- `scripts/build-engine.sh` has no HIP target; build with the `zig build` line in [HIP](HIP.md).
