#!/bin/bash

if [[ "${#}" > 0 ]]; then
  echo "USAGE:"
  echo "  ${0}"
  exit 1
fi

NUM_EPOCHS=30
DATASET="ogbn-products"

pushd examples/pytorch/graphsage/

DGL_USE_CUDA_MEMORY_POOL=1 OMP_NUM_THREADS=20 python3 train_sampling_multi_gpu.py --gpu="-1,-1" \
	--num-epochs "${NUM_EPOCHS}" \
	--dataset "${DATASET}" \
	--num-workers=4 \
	--num-hidden 256 \
	--fan-out=15,10,5 --num-layers=3 \
	--batch-size=2000

popd
