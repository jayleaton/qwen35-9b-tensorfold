# Known issues

## All devices

- **No speculative drafter beyond copy drafts.** The checkpoint ships an MTP head, but CUDA drafts with an MTP head
  only over an MoE target and the Metal `qwen27` engine has no MTP drafter (pins and upstream `main`); the converter
  leaves `mtp.*` out. [`z-lab/Qwen3.5-9B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) is a DFlash v1
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
