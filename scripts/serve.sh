#!/usr/bin/env bash
# Serve a Qwen3.5-9B tree (MLX affine or EXL3) with one backend's measured profile (config/<backend>/serve[-PROFILE].args).
#
#   scripts/serve.sh BACKEND ENGINE_DIR MODEL_DIR [extra tensorfold-native serve flags]
#
# PORT (default 8090) and NAME (default q9) set the listen port and the served model name; PROFILE=NAME picks
# config/<backend>/serve-NAME.args (cuda: batch; metal: 24gb, 16gb). A profile's .env file beside it (KEY=VALUE lines)
# is exported first; a variable already set in the environment wins. A profile with a .mtp file beside it drafts with
# the checkpoint's MTP head (--drafter MODEL_DIR) when MODEL_DIR holds mtp.safetensors (scripts/mtp-head.sh); MTP=0
# turns that off, MTP=1 asks for it on any profile.
set -euo pipefail
backend=${1:?usage: scripts/serve.sh BACKEND ENGINE_DIR MODEL_DIR [flags]}
engine=${2:?usage: scripts/serve.sh BACKEND ENGINE_DIR MODEL_DIR [flags]}
model=${3:?usage: scripts/serve.sh BACKEND ENGINE_DIR MODEL_DIR [flags]}
shift 3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ "$backend" =~ ^[a-z0-9-]+$ && -d "$root/config/$backend" ]] || { echo "no backend config/$backend" >&2; exit 2; }
args=$root/config/$backend/serve${PROFILE:+-$PROFILE}.args
[[ -f "$args" ]] || { echo "no profile $args" >&2; exit 2; }
read -r -a profile < "$args"
env_file=${args%.args}.env
if [[ -f "$env_file" ]]; then
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || { echo "bad line in $env_file: $key" >&2; exit 2; }
        [[ -n "${!key+x}" ]] || export "$key=$value"
    done < "$env_file"
fi
mtp=${MTP:-}
if [[ -z "$mtp" ]]; then
    [[ -f "${args%.args}.mtp" && -f "$model/mtp.safetensors" ]] && mtp=1 || mtp=0
fi
drafter=()
case "$mtp" in
    0) ;;
    1)
        [[ -f "$model/mtp.safetensors" ]] || { echo "$model has no mtp.safetensors: run scripts/mtp-head.sh" >&2; exit 2; }
        drafter=(--drafter "$model")
        ;;
    *) echo "MTP must be 0 or 1" >&2; exit 2 ;;
esac
lib=$engine/zig-out/native/lib
[[ -f "$lib/libtfgrammar.so" || -f "$lib/libtfgrammar.dylib" ]] || echo "warning: no libtfgrammar in $lib:" \
    "tool_choice required or named and response_format get HTTP 400; rebuild with scripts/build-engine.sh" >&2
exec "$engine/zig-out/native/bin/tensorfold-native" serve "$model" --name "${NAME:-q9}" --port "${PORT:-8090}" \
    "${profile[@]}" ${drafter[@]+"${drafter[@]}"} "$@"
