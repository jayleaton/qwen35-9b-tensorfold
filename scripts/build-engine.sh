#!/usr/bin/env bash
# Fetch and patch one backend's engine (scripts/fetch-engine.sh), then build tensorfold-native in it.
#
#   scripts/build-engine.sh cuda DEST [SOURCE]     on a DGX Spark; NVCC (default /usr/local/cuda/bin/nvcc) builds the kernels
#   scripts/build-engine.sh metal DEST [SOURCE]    on Apple Silicon with Xcode's macOS SDK and Metal toolchain
#   scripts/build-engine.sh rtx DEST [SOURCE]      an RTX card: the CUDA series plus EXL3, kernels for SM (default 120)
#   scripts/build-engine.sh upstream DEST [SOURCE] upstream main with no patches: Metal on macOS, CUDA elsewhere
#
# NVCC=none builds the CUDA side's host code without kernel images (a source check on a machine without CUDA).
# cuda and rtx also build libtfgrammar beside the server (tool_choice required or named, response_format) from xgrammar
# v0.2.8: XGRAMMAR=DIR names a checkout, otherwise DEST/.xgrammar is cloned (git clone --branch v0.2.8
# --recurse-submodules https://github.com/mlc-ai/xgrammar). XGRAMMAR=none skips it; those requests then get HTTP 400.
# On metal, XGRAMMAR=DIR builds it the same way.
set -euo pipefail
backend=${1:?usage: scripts/build-engine.sh cuda|metal|rtx|upstream DEST [SOURCE]}
dest=${2:?usage: scripts/build-engine.sh cuda|metal|rtx|upstream DEST [SOURCE]}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
"$root/scripts/fetch-engine.sh" "$@"
want=$(cat "$dest/.zig-version")
have=$(zig version)
[[ "$have" == "$want" ]] || { echo "zig $have found; the engine needs zig $want (.zig-version)" >&2; exit 2; }
cd "$dest"
[[ "$backend" == upstream ]] && { [[ "$(uname -s)" == Darwin ]] && backend=metal || backend=cuda; }
case "$backend" in
    cuda)
        nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
        if [[ "$nvcc" == none ]]; then
            zig build native
        else
            zig build native -Dnvcc="$nvcc"
            zig build qwen35-cuda -Dnvcc="$nvcc"
        fi
        ;;
    metal) zig build native -Dcpu=apple_m1 ;;
    rtx)
        nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
        zig build native -Dsm="${SM:-120}" -Dnvcc="$nvcc"
        zig build qwen35-cuda -Dsm="${SM:-120}" -Dnvcc="$nvcc"
        ;;
    *) echo "no build for $backend" >&2; exit 2 ;;
esac
xgrammar_commit=97787376faee5ed8466cfad57c99855e4ce2f6aa # v0.2.8
xg=${XGRAMMAR:-}
if [[ -z "$xg" && ( "$backend" == cuda || "$backend" == rtx ) ]]; then
    xg=$PWD/.xgrammar
    if [[ ! -d "$xg/.git" ]]; then
        git clone -q --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar "$xg.part"
        mv "$xg.part" "$xg"
    fi
    grep -qx '/.xgrammar' .git/info/exclude 2>/dev/null || echo '/.xgrammar' >> .git/info/exclude
fi
if [[ -n "$xg" && "$xg" != none ]]; then
    got=$(git -C "$xg" rev-parse HEAD)
    [[ "$got" == "$xgrammar_commit" ]] || { echo "$xg is xgrammar $got, not v0.2.8 ($xgrammar_commit)" >&2; exit 2; }
    if [[ "$backend" == metal ]]; then
        zig build grammar -Dcpu=apple_m1 -Dxgrammar="$(cd "$xg" && pwd)"
    else
        zig build grammar -Dxgrammar="$(cd "$xg" && pwd)"
    fi
    [[ -f zig-out/native/lib/libtfgrammar.so || -f zig-out/native/lib/libtfgrammar.dylib ]] ||
        { echo "zig build grammar wrote no libtfgrammar under zig-out/native/lib" >&2; exit 2; }
fi
echo "built $dest/zig-out/native/bin/tensorfold-native"
