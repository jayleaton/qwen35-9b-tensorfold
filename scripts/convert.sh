#!/usr/bin/env bash
# Convert the BF16 Qwen3.5-9B checkpoint to the engine's MLX affine format and give the tree its sampling defaults.
#
#   scripts/convert.sh ENGINE_DIR BF16_DIR OUT [BITS]     BITS: 4 (default) or 8, groups of 64
#
# ENGINE_DIR is the metal engine (scripts/fetch-engine.sh metal ENGINE_DIR: git only, no build, any OS). Its
# tools/sensenova_loop_mlx.py reads stock Qwen3.5 too, stores the gated norm in bf16 and drops the MTP layer, so the
# tree serves on the Spark (cuda or upstream main) and on the Macs alike. The checkpoint ships no
# generation_config.json, so without one the engine samples greedily; this writes Qwen's thinking-mode defaults
# (temperature 1.0, top_p 0.95) and both end tokens (<|endoftext|>, <|im_end|>). A request's own fields still win.
set -euo pipefail
engine=${1:?usage: scripts/convert.sh ENGINE_DIR BF16_DIR OUT [BITS]}
bf16=${2:?usage: scripts/convert.sh ENGINE_DIR BF16_DIR OUT [BITS]}
out=${3:?usage: scripts/convert.sh ENGINE_DIR BF16_DIR OUT [BITS]}
bits=${4:-4}
[[ "$bits" == 4 || "$bits" == 8 ]] || { echo "BITS must be 4 or 8" >&2; exit 2; }
[[ ! -e "$out" ]] || { echo "$out already exists" >&2; exit 2; }
grep -q 'stock Qwen3.5 ships it f32' "$engine/tools/sensenova_loop_mlx.py" 2>/dev/null ||
    { echo "$engine has no stock-Qwen3.5 converter: use scripts/fetch-engine.sh metal DIR" >&2; exit 2; }
python3 "$engine/tools/sensenova_loop_mlx.py" --source "$bf16" --out "$out" --bits "$bits"
cat > "$out/generation_config.json" <<'EOF'
{
  "do_sample": true,
  "eos_token_id": [248044, 248046],
  "temperature": 1.0,
  "top_p": 0.95
}
EOF
echo "converted $out: MLX affine ${bits}-bit g64, sampling temperature 1.0 / top_p 0.95"
