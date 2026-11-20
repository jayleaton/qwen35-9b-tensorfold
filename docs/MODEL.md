# The checkpoint

[`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`,
BF16, four shards (19.3 GB), Apache-2.0. `Qwen3_5ForConditionalGeneration`: a text decoder, a vision tower and one MTP
layer.

| | |
| --- | --- |
| Decoder | 32 layers, hidden 4096, MLP 12,288, vocabulary 248,320 |
| Attention | hybrid: every fourth layer is full attention (8 layers: 16 query heads, 4 KV heads, head 256, output gate), the other 24 are Gated DeltaNet (16 key / 32 value heads of 128, conv kernel 4) |
| Positions | 262,144 (`max_position_embeddings`); M-RoPE, theta 10,000,000, a quarter of each head rotated |
| MTP | one layer (`mtp_num_hidden_layers: 1`, 15 `mtp.*` tensors) |
| Vision | 27 blocks, width 1152, merged to 4096 |
| Sampling | no `generation_config.json` in the repository: an engine without one samples greedily |

## Cache sizing

Only the 8 full-attention layers keep a KV cache, so a token costs 8 × 2 × 4 heads × 256 × 1 byte = **16 KiB with fp8
KV** (32 KiB in bf16); the engine's banner reports 16.25 KiB with page padding. A full 262,144-token window is about
4.2 GiB of fp8 KV a stream. The 24 Gated DeltaNet layers keep a fixed recurrent state per stream, independent of
length.

## Memory

| tree | on disk | engine-reported model size |
| --- | ---: | ---: |
| `q9-q4` (MLX affine 4-bit g64, vision tower carried in bf16) | 5.96 GB | 5.27 GiB on CUDA (text only) |
| `q9-q8` (8-bit g64) | 10.44 GB | 9.44 GiB on CUDA |
| EXL3 4.0 / 5.0 / 6.0 bpw ([RTX](RTX.md)) | 6.30 / 7.16 / 8.28 GB | 6.43 GiB (4.0), 8.28 GiB (6.0) |

## Drafting

The checkpoint ships an MTP head, but none of the engines in this recipe drafts with a dense target's MTP head (CUDA
opens MTP only over an MoE target; the Metal `qwen27` engine has no MTP drafter). An upstream draft PR's drafter was
measured with it: [CUDA](CUDA.md#follow-up-mtp-drafting). Copy drafts, which draft from a
chat's own context and need no extra weights, are on by default everywhere. See [KNOWN-ISSUES](KNOWN-ISSUES.md).
