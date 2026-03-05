#!/bin/bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
export LD_PRELOAD=/data/miniconda3/envs/social/lib/libstdc++.so.6.0.33

source /data/miniconda3/etc/profile.d/conda.sh
conda activate social
exec python $(rospack find path_select)/nodes/sequencer.py "$@"

