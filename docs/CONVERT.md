# Convert BF16 to MLX affine

On CUDA, TensorFold serves a converted checkpoint (a plain BF16 Qwen3.5 tree is refused for serving), and the Metal
profile serves 4-bit too. One converted tree serves every MLX device in this recipe: the DGX Spark (`cuda`, and
upstream main) and the Macs (`metal`).

The converter is `tools/sensenova_loop_mlx.py` from the `metal` series (written for SenseNova-Loop-9B, it reads stock
Qwen3.5 too). Fetching the `metal` engine is a git operation, so it works on Linux as well; nothing is built:

```bash
pip install numpy torch safetensors          # CPU torch is enough
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir Qwen3.5-9B
scripts/fetch-engine.sh metal engine-metal   # on a Mac, scripts/build-engine.sh metal engine-metal does this too
scripts/convert.sh engine-metal Qwen3.5-9B q9-q4 4       # or 8 for q9-q8
```

Each width takes about a minute on a DGX Spark's CPU. The converter reads one tensor at a time, writes 1.8 GB shards
plus `model.safetensors.index.json`, copies the tokenizer and chat template, and records a round-trip check of its
first quantized tensors in `tensorfold-convert.json`.

**Use the `metal` copy, not the one in the `cuda` tree.** The `cuda` series carries an older copy (`patches/cuda/0008`)
that keeps `linear_attn.norm` in f32 and keeps the MTP layer. The patched CUDA loader widens that norm (`cuda` 0004)
and greedy replies are token-identical to the `metal` copy's tree, but upstream TensorFold refuses it
(`UnexpectedTensor`), and the tree is then no longer the one every device reads. `scripts/convert.sh` refuses an
engine directory without the stock-Qwen3.5 converter.

## The sampling file

The checkpoint ships no `generation_config.json`, so without one the engine samples greedily. `scripts/convert.sh`
writes Qwen's thinking-mode defaults and both end tokens:

```json
{"do_sample": true, "eos_token_id": [248044, 248046], "temperature": 1.0, "top_p": 0.95}
```

The server banner then reads `sampling: temperature 1.0, top_p 0.95`. A request's own `temperature` and `top_p` still
win; the benchmark spot checks send `temperature: 0`.

## What it writes

| source | written as |
| --- | --- |
| 2-D projections, `embed_tokens`, `lm_head` | MLX affine codes: `u32` words, `bf16` scales and biases per group of 64 |
| `input_layernorm`, `post_attention_layernorm`, `model.norm`, `q_norm`, `k_norm` | `1 + w` in bf16 (`Qwen3_5RMSNorm` stores a zero-centred weight) |
| `linear_attn.norm` (f32 in the checkpoint) | bf16 (`metal` 0037) |
| `linear_attn.conv1d.weight` `[qkv, 1, kernel]` | `[qkv, kernel, 1]` |
| `A_log`, `dt_bias` (f32) | bf16 |
| `model.visual.*` | `vision_tower.*`, 333 tensors in bf16, unquantized; the CUDA text engine does not load them |
| `mtp.*` | left out (`metal` 0035): no engine in the recipe drafts with a dense MTP head ([CUDA](CUDA.md#follow-up-mtp-drafting)) |

| tree | tensors | on disk |
| --- | ---: | ---: |
| `q9-q4` | 1,260 | 5.96 GB |
| `q9-q8` | 1,260 | 10.44 GB |
