# The engine: pins and patch series

Each config builds upstream [TensorFold](https://github.com/ashhart/TensorFold) at an exact commit, with a patch
series applied in order where it has one. The pin, the series and the resulting tree are recorded in `config/<config>/`:

| config | pin (`tensorfold.rev`) | series (`tensorfold.series` / `.patches`) | `tensorfold.tree` | devices |
| --- | --- | --- | --- | --- |
| `cuda` | `038e4a27872eaa018adc67b1b5ddffd16820c498` | `cuda:0027` | `9bac6ecbdef1c03861d824e930799edcb3b3de16` | DGX Spark |
| `rtx` | same | `cuda:0022 rtx:0003` | `64ba7d4085cc4742c127af91fa196218b189bdbf` | 16 GB RTX |
| `metal` | `6d0a128b0cde35aa091bda933577cfced3519b8a` | `metal` `0001`-`0044` | `d33e2d93ea37f0b3ed1f208599a24ac00b3de16e` | Macs |
| `hip` | `813ff4ccbf9d6610e991a96d53fda2b628f56347` | none | `f70d190f1d1bd65575a53f7eb06838c7acf97cfc` | Ally X ([HIP](HIP.md)) |
| `upstream` | `727153cb33876173efb256d60a12e69413cf1248` | none | `c4e66e4220a8c650d1d94b0fdda4906d6dced988` | control |

Every repo is `https://github.com/ashhart/TensorFold.git`. `scripts/fetch-engine.sh CONFIG DEST [SOURCE]` fetches the
pin, applies each `DIR:LAST` series in order with `git am` (each patch keeps its author; committer and dates are fixed,
so every machine gets the same commits) and refuses any result whose tree differs from `tensorfold.tree`. SOURCE may
be a local TensorFold checkout or a bundle that holds the pin (useful on a machine without GitHub access).
`scripts/build-engine.sh` then builds `tensorfold-native` (and `tf-qwen35-cuda` on CUDA); with `XGRAMMAR=DIR` it also
builds `libtfgrammar`.

## Why these series for a stock checkpoint

The series were written for [SenseNova-Loop-9B](https://github.com/jayleaton/sensenova-loop9b-tensorfold), a looped
Qwen3.5-9B. They are reused byte for byte (`rtx` `0003` is the one patch added here), so the `cuda` and `metal` trees
are ones already checked there. Their loop and mHC code runs only for a checkpoint with `loop_settings`; stock
Qwen3.5-9B has none and takes the plain `qwen3_5` path. What the stock model uses from them:

| patches | what stock Qwen3.5-9B gets | on upstream `main` (`727153cb3`) |
| --- | --- | --- |
| `cuda` `0004` | an f32 `linear_attn.norm` widened at load | refused: `UnexpectedTensor` (see [CONVERT](CONVERT.md)) |
| `cuda` `0013` | a required or named tool call closes the think block in time for the call | applies cleanly, so not there |
| `cuda` `0014` | `qwen3_5` CUDA takes a 0.5 decode share by default | conflicts (`cuda.zig` refactored) |
| `cuda` `0015`, `0016` | `--spill-gib`'s evictions are not charged to the daily write budget; the disk tier's staging permit fits the longest kept state | `0015` applies cleanly; `0016` conflicts (`cuda_disk.zig` refactored) |
| `cuda` `0017`-`0022` | copy drafts in shared rounds: windows start narrow and widen as copies land, and copy rows fill the backend's 16-row tile split by acceptance | `0017`-`0019` apply cleanly; `0020`-`0022` conflict |
| `cuda` `0024`-`0027` | the one-barrier kv8 prompt attention (same bits) | not there |
| `rtx` `0001`, `0002` | the EXL3 capability list compiles and keeps Qwen's decisions; `cuda.zig` back to 600 lines | not there |
| `rtx` `0003` | new for this recipe, tools only: `exl3_attach_sensenova.py attach --method` records `exllamav3-native` calibration ([RTX](RTX.md#quantize)) | not there |
| `cuda` `0008`, `metal` `0002`, `0004`, `0035`, `0037` | the converter, `tools/sensenova_loop_mlx.py` | no Qwen3.5 converter in `tools/` |
| `metal` `0034` | a startup margin that scales with the working set (16 GB Macs) | |
| `metal` `0040`-`0043` | macOS 26 (MSL 4.0) compiles the library; lone streams propose copy drafts; copy drafts on by default | |

`cuda` `0001`-`0003`, `0005`-`0007`, `0009`-`0012` and `0023` are the loop, the mHC and their image path: present in the
tree, idle for this checkpoint. The full patch-by-patch list is in the
[SenseNova recipe's ENGINE](https://github.com/jayleaton/sensenova-loop9b-tensorfold/blob/main/docs/ENGINE.md).

## The `upstream` control

`config/upstream/` is upstream `main` with no patches, built and served on the same DGX Spark with the same flags, to
show what the series add. Its results are in [CUDA](CUDA.md#upstream-main). It is not a supported config.
