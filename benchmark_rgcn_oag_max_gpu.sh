#!/bin/bash

if [[ "${#}" != 1 ]]; then
  echo "USAGE:"
  echo "  ${0} <gpu list>"
  echo "    Where <gpu list> is a comma separated list of GPUs to use."
  echo "    For eample, to run on all 8 GPUs:"
  echo "      ${0} 0,1,2,3,4,5,6,7"
  exit 1
fi

GPUS="${1}"
NUM_EPOCHS=30
DATASET="oag_max.dgl"

pushd examples/pytorch/rgcn/

DGL_USE_CUDA_MEMORY_POOL=1 OMP_NUM_THREADS=6 ${NSYS} python3 entity_classify_mp.py --gpu="${GPUS}" \
  -d "${DATASET}" \
  --testing --fanout='25,30' --batch-size 2000 --n-hidden 64 --lr 0.01 \
  --num-worker 6 --eval-batch-size 8 --low-mem --gpu="${GPUS}" --dropout 0.5 \
  --use-self-loop --n-bases 2 --n-epochs "${NUM_EPOCHS}" --mix-cpu-gpu --node-feats

popd
