#!/bin/bash

if [[ "${#}" != 1 ]]; then
  echo "USAGE:"
  echo "  ${0} <gpu list>"
  echo ""
  echo "  Where <gpu list> is a comma separated list of GPUs to use."
  echo "  For eample, to run on all 8 GPUs:"
  echo "    ${0} 0,1,2,3,4,5,6,7"
  exit 1
fi



GPUS="${1}"
NUM_EPOCHS=30
DATASET="ogbn-products"

pushd examples/pytorch/graphsage/

DGL_USE_CUDA_MEMORY_POOL=1 OMP_NUM_THREADS=6 python3 train_sampling_multi_gpu.py --gpu="${GPUS}" \
	--num-epochs "${NUM_EPOCHS}" \
	--dataset "${DATASET}" \
	--num-workers=6 \
	--num-hidden 256 \
	--fan-out=15,10,5 --num-layers=3 \
	--batch-size=2000

popd
