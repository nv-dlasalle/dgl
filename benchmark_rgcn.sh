#!/bin/bash

GPUS="${1}"
NUM_EPOCHS=30
DATASET="oag_max.dgl"

pushd examples/pytorch/rgcn/

CUDA_MEMCHECK_PATCH_MODULE=1 DGL_USE_CUDA_MEMORY_POOL=1 OMP_NUM_THREADS=6 ${NSYS} python3 entity_classify_mp.py --gpu="${GPUS}" \
  -d "${DATASET}" \
  --testing --fanout='25,30' --batch-size 2000 --n-hidden 64 --lr 0.01 \
  --num-worker 6 --eval-batch-size 8 --low-mem --gpu="${GPUS}" --dropout 0.5 \
  --use-self-loop --n-bases 2 --n-epochs "${NUM_EPOCHS}" --mix-cpu-gpu --node-feats

popd
