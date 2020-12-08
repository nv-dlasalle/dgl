#!/bin/bash

GPUS="${1}"

#NSYS="nsys profile -o /tmp/graphsage_2gpu -f true --trace=cuda,nvtx --trace-fork-before-exec=true"

DGL_USE_CUDA_MEMORY_POOL=1 OMP_NUM_THREADS=6 ${NSYS} python3 train_sampling_multi_gpu.py --num-epochs 4 --dataset ogbn-products --gpu ${GPUS} --num-workers=6
