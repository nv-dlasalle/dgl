#!/bin/bash

GPUS="${1}"

#NSYS="nsys profile -o /tmp/rgcn_base -f true --trace=nvtx,cuda -s none -b none"

DGL_USE_CUDA_MEMORY_POOL=1 OMP_NUM_THREADS=6 ${NSYS} python3 entity_classify_mp.py -d ogbn-mag \
  --testing --fanout='25,30' --batch-size 512 --n-hidden 64 --lr 0.01 \
  --num-worker 6 --eval-batch-size 8 --low-mem --gpu ${GPUS} --dropout 0.5 \
  --use-self-loop --n-bases 2 --n-epochs 3 --mix-cpu-gpu --node-feats \
  --sparse-embedding
