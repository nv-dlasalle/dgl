#!/bin/bash

GPUS="${1}"

#NSYS="nsys profile -o /tmp/graphsage_2gpu -f true --trace=cuda,nvtx --trace-fork-before-exec=true"

DGL_USE_CUDA_MEMORY_POOL=1 OMP_NUM_THREADS=6 ${NSYS} python3 train_sampling_multi_gpu.py --gpu ${GPUS} \
	--num-epochs 4 \
	--dataset ogbn-papers100M \
	--num-workers=6 \
	--num-hidden 256 \
	--fan-out=15,10,5 --num-layers=3 \
	--batch-size=2000
