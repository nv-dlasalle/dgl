#!/bin/bash

if [[ "${#}" > 0 ]]; then
  echo "USAGE:"
  echo "  ${0}"
  exit 1
fi


CUM_EPOCHS=30
DATASET="oag_min.dgl"

# NOTE: this is configured for a system with 2 sockets, each with 24 physical
# cores.

pushd examples/pytorch/rgcn/

DGL_USE_CUDA_MEMORY_POOL=1 OMP_NUM_THREADS=20 ${NSYS} python3 entity_classify_mp.py --gpu="-1,-1" \
  -d "${DATASET}" \
  --testing --fanout='25,30' --batch-size 2000 --n-hidden 64 --lr 0.01 \
  --num-worker 4 --eval-batch-size 8 --low-mem --gpu="${GPUS}" --dropout 0.5 \
  --use-self-loop --n-bases 2 --n-epochs "${NUM_EPOCHS}" --mix-cpu-gpu --node-feats

popd
