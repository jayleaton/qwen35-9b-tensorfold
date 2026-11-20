#!/usr/bin/env bash
# Quantize the BF16 checkpoint to an EXL3 pack with exllamav3 at the pinned commit, for the rtx engine.
#
#   scripts/quantize-exl3.sh ENGINE_DIR BF16_DIR OUT [BITS]     ENGINE_DIR from scripts/fetch-engine.sh rtx; BITS 6.0
#
# BITS names a recipe in ENGINE_DIR/docs/exl3-recipes/ (4.0, 5.0 or 6.0). exllamav3 traces stock qwen3_5 itself, so it
# calibrates in its own forward (250 x 2048 rows); the engine's shim and attach tools wrap it. X3_DIR (default
# ./exllamav3) is the exllamav3 checkout and X3_VENV (default ./x3venv) its Python; both are created when missing.
# NVCC (default /usr/local/cuda/bin/nvcc) gives CUDA_HOME for exllamav3's extension; SM (default 120) its target.
# Work files go to OUT.work; a run stopped part-way resumes from them.
set -euo pipefail
engine=${1:?usage: scripts/quantize-exl3.sh ENGINE_DIR BF16_DIR OUT [BITS]}
src=${2:?usage: scripts/quantize-exl3.sh ENGINE_DIR BF16_DIR OUT [BITS]}
out=${3:?usage: scripts/quantize-exl3.sh ENGINE_DIR BF16_DIR OUT [BITS]}
bits=${4:-6.0}
x3_rev=8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0
attach=$engine/tools/exl3_attach_sensenova.py
recipe=$engine/docs/exl3-recipes/$bits.json
grep -q -- '--method' "$attach" || { echo "$engine lacks rtx 0003; use scripts/fetch-engine.sh rtx DIR" >&2; exit 2; }
[[ -f "$recipe" ]] || { echo "no recipe $recipe" >&2; exit 2; }
[[ -f "$src/config.json" ]] || { echo "no checkpoint in $src" >&2; exit 2; }
engine=$(cd "$engine" && pwd) src=$(cd "$src" && pwd)
mkdir -p "$out.work" && out=$(cd "$(dirname "$out")" && pwd)/$(basename "$out")
x3=${X3_DIR:-$PWD/exllamav3}
venv=${X3_VENV:-$PWD/x3venv}
py=$venv/bin/python
if [[ ! -d "$x3/.git" ]]; then
    git clone -q https://github.com/turboderp-org/exllamav3 "$x3"
fi
git -C "$x3" checkout -q --detach "$x3_rev"
if [[ ! -x "$py" ]]; then
    python3 -m venv "$venv"
    "$py" -m pip install -q --upgrade pip wheel setuptools ninja
    "$py" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu130
    "$py" -m pip install -q -r "$x3/requirements.txt"
fi
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
sm=${SM:-120}
cuda_home=$(cd "$(dirname "$nvcc")/.." && pwd)
nv=$("$py" -I -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))')
# exllamav3's extension needs cuBLAS and cuSPARSE headers; torch's cu13 wheels carry them when the toolkit does not.
export CUDA_HOME=$cuda_home PATH=$venv/bin:$cuda_home/bin:$PATH TORCH_CUDA_ARCH_LIST=${sm:0:-1}.${sm: -1}
export CPATH=$nv/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$nv/lib:$cuda_home/lib64:$cuda_home/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=$nv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
read -r b hb eb cb scales < <("$py" -I -c 'import json, sys; r = json.load(open(sys.argv[1]))
print(r["bits"], r["head_bits"], r["embed_bits"], r["codebook"], r["out_scales"])' "$recipe")
[[ -e "$out.work/shim/config.json" ]] || "$py" -I "$attach" shim --source "$src" --out "$out.work/shim"
if [[ -d "$out.work/convert/ckpt" ]]; then
    (cd "$x3" && "$py" convert.py -w "$out.work/convert" -r)
else
    (cd "$x3" && "$py" convert.py -i "$out.work/shim" -o "$out" -w "$out.work/convert" -b "$b" -hb "$hb" -eb "$eb" \
        -cb "$cb" --out_scales "$scales" -d 0)
fi
"$py" -I "$attach" attach --source "$src" --converted "$out" --recipe "st-$bits" --method exllamav3-native
cat > "$out/generation_config.json" <<'EOF'
{"do_sample": true, "eos_token_id": [248044, 248046], "temperature": 1.0, "top_p": 0.95}
EOF
echo "wrote $out; the work files in $out.work can go"
