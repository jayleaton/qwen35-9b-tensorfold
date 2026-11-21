#!/usr/bin/env bash
# Save the checkpoint's MTP head beside a converted tree, so a cuda build drafts with it (--drafter TREE).
#
#   scripts/mtp-head.sh BF16_DIR TREE
#
# Writes TREE/mtp.safetensors: the checkpoint's 15 mtp.* tensors, their BF16 bytes copied as shipped (486,582,880
# bytes; Python's standard library only). The tree's shards and model.safetensors.index.json are not touched, so an
# engine that does not draft with MTP reads exactly the tree it read before; the cuda build's MTP loader finds the file
# beside the index and quantizes the head to affine-4 tables at load.
set -euo pipefail
bf16=${1:?usage: scripts/mtp-head.sh BF16_DIR TREE}
tree=${2:?usage: scripts/mtp-head.sh BF16_DIR TREE}
[[ -f "$bf16/model.safetensors.index.json" ]] || { echo "$bf16 has no model.safetensors.index.json" >&2; exit 2; }
[[ -f "$tree/model.safetensors.index.json" ]] || { echo "$tree is not a converted tree" >&2; exit 2; }
[[ ! -e "$tree/mtp.safetensors" ]] || { echo "$tree/mtp.safetensors already exists" >&2; exit 2; }
python3 -I - "$bf16" "$tree" <<'PY'
import json
import os
import struct
import sys

bf16, tree = sys.argv[1:3]
weight_map = json.load(open(os.path.join(bf16, "model.safetensors.index.json")))["weight_map"]
if any(k.startswith("mtp.") for k in json.load(open(os.path.join(tree, "model.safetensors.index.json")))["weight_map"]):
    sys.exit(f"{tree}'s index already lists mtp.* tensors")
names = sorted(k for k in weight_map if k.startswith("mtp."))
if len(names) != 15:
    sys.exit(f"{bf16}: expected 15 mtp.* tensors, found {len(names)}")
header, chunks, at = {}, [], 0
for name in names:
    with open(os.path.join(bf16, weight_map[name]), "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        entry = json.loads(f.read(size))[name]
        if entry["dtype"] != "BF16":
            sys.exit(f"{name} is {entry['dtype']}, not BF16")
        lo, hi = entry["data_offsets"]
        f.seek(8 + size + lo)
        chunks.append(f.read(hi - lo))
    header[name] = {"dtype": "BF16", "shape": entry["shape"], "data_offsets": [at, at + hi - lo]}
    at += hi - lo
header["__metadata__"] = {"format": "pt"}
text = json.dumps(header, separators=(",", ":")).encode()
text += b" " * (-len(text) % 8)
out = os.path.join(tree, "mtp.safetensors")
with open(out + ".part", "wb") as f:
    f.write(struct.pack("<Q", len(text)) + text)
    for chunk in chunks:
        f.write(chunk)
os.replace(out + ".part", out)
print(f"{out}: {len(names)} mtp.* tensors, {os.path.getsize(out):,} bytes")
PY
