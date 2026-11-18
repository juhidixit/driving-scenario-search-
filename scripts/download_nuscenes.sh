#!/usr/bin/env bash
# Download the parts of nuScenes v1.0-trainval this project uses (~46 GB transferred, ~6 GB kept).
#
# nuScenes is released under CC BY-NC-SA 4.0. Create an account and accept the terms of use at
# https://www.nuscenes.org/nuscenes before downloading. Nothing from the dataset is committed to
# this repository.
set -euo pipefail
ROOT="${NUSCENES_ROOT:-data/nuscenes}"
BASE="${NUSCENES_URL:-https://motional-nuscenes.s3.amazonaws.com/public/v1.0}"
mkdir -p "$ROOT" && cd "$ROOT"

fetch() {  # resumable download with retries
  for i in $(seq 1 20); do curl -fsS -C - -o "$1" "$BASE/$1" && return 0; echo "retry $1 ($i)"; sleep 10; done
  return 1
}

[ -d v1.0-trainval ] || { fetch v1.0-trainval_meta.tgz && tar xzf v1.0-trainval_meta.tgz && rm v1.0-trainval_meta.tgz; }
[ -d can_bus ] || { fetch can_bus.zip && unzip -qo can_bus.zip && rm can_bus.zip; }
[ -d maps/expansion ] || { fetch nuScenes-map-expansion-v1.3.zip && unzip -qo nuScenes-map-expansion-v1.3.zip -d maps && rm nuScenes-map-expansion-v1.3.zip; }

# Key-frame camera images: keep CAM_FRONT only (the other cameras and lidar are skipped).
if [ "${SKIP_IMAGES:-0}" != "1" ]; then
  for i in 01 02 03 04 05 06 07 08 09 10; do
    [ -f ".done_kf_$i" ] && continue
    f="v1.0-trainval${i}_keyframes.tgz"
    fetch "$f" && tar xzf "$f" --wildcards 'samples/CAM_FRONT/*' && rm "$f" && touch ".done_kf_$i"
  done
fi
echo "nuScenes ready in $ROOT"
