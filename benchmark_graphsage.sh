#!/bin/bash

GPUS="${1}"
NUM_EPOCHS=30

pushd examples/pytorch/graphsage/

DGL_USE_CUDA_MEMORY_POOL=1 OMP_NUM_THREADS=6 python3 train_sampling_multi_gpu.py --gpu="${GPUS}" \
	--num-epochs "${NUM_EPOCHS}" \
	--dataset ogbn-products \
	--num-workers=6 \
	--num-hidden 256 \
	--fan-out=15,10,5 --num-layers=3 \
	--batch-size=2000

popd
