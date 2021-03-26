/*!
 *  Copyright (c) 2021 by Contributors
 * \file nccl_api.cc
 * \brief Implementation of wrapper around NCCL routines. 
 */


#ifndef DGL_RUNTIME_CUDA_NCCL_API_H_
#define DGL_RUNTIME_CUDA_NCCL_API_H_

#include "nccl_api.h"
#include "cuda_common.h"
#include "../../kernel/cuda/atomic.cuh"
#include "../../array/cuda/dgl_cub.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/device_api.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>

#define NCCL_CALL(func) \
{ \
  ncclResult_t result = func; \
  if (result != ncclSuccess) { \
      LOG(FATAL)                                                        \
          << "NCCLError: " #func " failed with error: " << result;            \
  } \
}

namespace dgl {

using namespace kernel::cuda;

namespace runtime {
namespace cuda {

namespace {

enum class AllToAllMode : int {
  REMAINDER = 0
};


template<typename T> ncclDataType_t NCCLType();
template<> ncclDataType_t NCCLType<int32_t>() {
    return ncclInt32; 
}
template<> ncclDataType_t NCCLType<int64_t>() {
    return ncclInt64; 
}
template<> ncclDataType_t NCCLType<__half>() {
    return ncclHalf; 
}
template<> ncclDataType_t NCCLType<float>() {
    return ncclFloat32; 
}
template<> ncclDataType_t NCCLType<double>() {
    return ncclFloat64; 
}


template<typename IdType> __global__ void _MapProcByRemainder(
    const IdType * const index,
    const int64_t num_index,
    const int64_t num_proc,
    IdType * const proc_id)
{
  const int64_t idx = blockDim.x*static_cast<int64_t>(blockIdx.x)+threadIdx.x;

  if (idx < num_index) {
    proc_id[idx] = index[idx] % num_proc;
  }
}

template<typename IdType>
__global__ void _MapProcByMaskRemainder(
    const IdType * const index,
    const int64_t num_index,
    const IdType mask,
    IdType * const proc_id)
{
  const int64_t idx = blockDim.x*static_cast<int64_t>(blockIdx.x)+threadIdx.x;

  if (idx < num_index) {
    proc_id[idx] = index[idx] & mask;
  }
}

template<typename IdType, typename DType>
__global__ void _DualPermKernel(
    const IdType * const in_idx,
    const DType * const in_value,
    const IdType * const perm,
    const int64_t num_in,
    const int64_t num_feat,
    IdType * const out_idx,
    DType * const out_value)
{
  // set index permutation
  const int64_t tidx = blockDim.x*static_cast<int64_t>(blockIdx.x)+threadIdx.x;
  if (tidx < num_in) {
    const IdType perm_idx = perm[tidx];
    assert(perm_idx < num_in);
    out_idx[perm_idx] = in_idx[tidx];
  }

  if (num_feat > 1) {
    for (int d = 0; d < blockDim.x; ++d) {
      const int64_t bidx = blockDim.x*static_cast<int64_t>(blockIdx.x) + d;
      if (bidx < num_in) {
        const IdType perm_idx = perm[bidx];
        for (int64_t f = threadIdx.x; f < num_feat; f+=blockDim.x) {
          out_value[perm_idx*num_feat+f] = in_value[bidx*num_feat+f];
        }
      }
    }
  } else {
    if (tidx < num_in) {
      const IdType perm_idx = perm[tidx];
      out_value[perm_idx] = in_value[tidx];
    }
  }
}

template<typename IdType, int MAX_BINS, int BLOCK_SIZE, int TILE_SIZE>
__global__ void _CountIndexByRemainder(
    const IdType * const items,
    const int64_t num_items,
    int64_t * const counts,
    const int num_counts)
{
  constexpr const int VALS_PER_THREAD = TILE_SIZE/BLOCK_SIZE;

  typedef cub::BlockHistogram<IdType, BLOCK_SIZE, VALS_PER_THREAD, MAX_BINS> BlockHistogram;

  __shared__ IdType local_counts[MAX_BINS];
  __shared__ typename BlockHistogram::TempStorage temp_storage;
  IdType thread_vals[VALS_PER_THREAD];

  const int64_t offset = TILE_SIZE*blockIdx.x;

  assert(num_counts < MAX_BINS);

  #pragma unroll
  for (int i = 0; i < VALS_PER_THREAD; ++i) {
    const int64_t in_idx = offset+threadIdx.x+(i*BLOCK_SIZE);
    thread_vals[i] = in_idx < num_items ? (items[in_idx] % num_counts): MAX_BINS-1;
  }

  BlockHistogram(temp_storage).Histogram(thread_vals, local_counts);

  __syncthreads();

  // write local histogram back to global memory
  for (int i = threadIdx.x; i < num_counts; i+=BLOCK_SIZE) {
    const int64_t val = local_counts[i];
    if (val > 0) {
      AtomicAdd(counts+i, val);
    }
  }
}

}

/* NCCLUniqueId **************************************************************/

NCCLUniqueId::NCCLUniqueId() :
  id_()
{
  // this ID is unique to the process, not to each call of this function
  NCCL_CALL(ncclGetUniqueId(&id_));
}

ncclUniqueId NCCLUniqueId::Get() const
{
  return id_;
}


/* NCCLCommunicator **********************************************************/

NCCLCommunicator::NCCLCommunicator(
    const int size,
    const int rank,
    ncclUniqueId id) :
  comm_(),
  size_(size),
  rank_(rank)
{
  CHECK_LT(rank, size);
  CHECK_GE(rank, 0);

  NCCL_CALL(ncclCommInitRank(&comm_, size_, id, rank_));
}

NCCLCommunicator::~NCCLCommunicator()
{
  ncclCommDestroy(comm_);
}

ncclComm_t NCCLCommunicator::Get()
{
  return comm_;
}

void NCCLCommunicator::AllToAllV(
    const void * const * const send,
    const int64_t * send_size,
    void * const * const recv,
    const int64_t * recv_size,
    const ncclDataType_t type,
    cudaStream_t stream)
{ 
  NCCL_CALL(ncclGroupStart());
  for (int r = 0; r < size_; ++r) {
    if (send_size[r] > 0) {
      NCCL_CALL(ncclSend(send[r], send_size[r], type, r, comm_, stream));
    }
    if (recv_size[r] > 0) {
      NCCL_CALL(ncclRecv(recv[r], recv_size[r], type, r, comm_, stream));
    }
  }
  NCCL_CALL(ncclGroupEnd());
}

template<typename IdType>
void NCCLCommunicator::AllToAll(
    const IdType * const send,
    IdType * const recv,
    const int64_t count,
    cudaStream_t stream)
{
  const ncclDataType_t type = NCCLType<IdType>();

  ncclGroupStart();
  for (int r = 0; r < size_; ++r) {
    ncclSend(send+(r*count), count, type, r, comm_, stream);
    ncclRecv(recv+(r*count), count, type, r, comm_, stream);
  }
  ncclGroupEnd();
}

template
void NCCLCommunicator::AllToAll<int32_t>(
    const int32_t * const send,
    int32_t * const recv,
    const int64_t count,
    cudaStream_t stream);
template
void NCCLCommunicator::AllToAll<int64_t>(
    const int64_t * const send,
    int64_t * const recv,
    const int64_t count,
    cudaStream_t stream);


template<typename IdType, typename DType>
void NCCLCommunicator::SparseAllToAll(
      const IdType * const send_idx,
      const DType * const send_value,
      const int64_t num_feat,
      const int64_t * const send_prefix,
      IdType * const recv_idx,
      DType * const recv_value,
      const int64_t * const recv_prefix,
      cudaStream_t stream)
{
  const ncclDataType_t idx_type = NCCLType<IdType>();
  const ncclDataType_t value_type = NCCLType<DType>();

  ncclGroupStart();
  for (int r = 0; r < size_; ++r) {
    const int64_t send_size = send_prefix[r+1]-send_prefix[r];
    if (send_size > 0) {
      ncclSend(send_idx+send_prefix[r], send_size, idx_type, r, comm_, stream);
      ncclSend(send_value+send_prefix[r]*num_feat, send_size*num_feat, value_type, r, comm_, stream);
    }
    const int64_t recv_size = recv_prefix[r+1]-recv_prefix[r];
    if (recv_size > 0) {
      ncclRecv(recv_idx+recv_prefix[r], recv_size, idx_type, r, comm_, stream);
      ncclRecv(recv_value+recv_prefix[r]*num_feat, recv_size*num_feat, value_type, r, comm_, stream);
    }
  }
  ncclGroupEnd();
}

template
void NCCLCommunicator::SparseAllToAll<int32_t, __half>(
      const int32_t * const send_idx,
      const __half * const send_value,
      const int64_t num_feat,
      const int64_t * const send_prefix,
      int32_t * const recv_idx,
      __half * const recv_value,
      const int64_t * const recv_prefix,
      cudaStream_t stream);

template
void NCCLCommunicator::SparseAllToAll<int64_t, __half>(
      const int64_t * const send_idx,
      const __half * const send_value,
      const int64_t num_feat,
      const int64_t * const send_prefix,
      int64_t * const recv_idx,
      __half * const recv_value,
      const int64_t * const recv_prefix,
      cudaStream_t stream);

template<typename IdType, typename DType>
void GenerateSparseBuffersFromRemainder(
    DeviceAPI* const device,
    const DGLContext& ctx,
    const int64_t comm_size,
    const int64_t num_in,
    const int64_t num_feat,
    const IdType * const in_idx,
    const DType * const in_value,
    IdType * const out_idx,
    DType * const out_value,
    int64_t * const out_counts,
    cudaStream_t stream)
{
  const int64_t comm_bits =
      static_cast<int64_t>(std::ceil(std::log2(comm_size)));

  // this should only run when we have things to send, otherwise comm_bits
  // will be zero, and several operations will fail
  CHECK_GT(comm_size, 1);

  CUDA_CALL(cudaMemsetAsync(
      out_counts, 0, sizeof(*out_counts)*(comm_size+1), stream));

  if (num_in == 0) {
    // now that we've zero'd out_counts, nothing left to do
    return;
  }

  // First, generate a mapping of indexes to processors
  IdType * proc_id_in = static_cast<IdType*>(
      device->AllocWorkspace(ctx, sizeof(IdType)*num_in));
  {
    const dim3 block(256);
    const dim3 grid((num_in+block.x-1)/block.x);

    if (comm_size < (1 << comm_bits)) {
      // comm_size is not a power of 2
      _MapProcByRemainder<<<grid, block, 0, stream>>>(
          in_idx,
          num_in,
          comm_size,
          proc_id_in);
      CUDA_CALL(cudaGetLastError());
    } else {
      // comm_size is a power of 2
      _MapProcByMaskRemainder<<<grid, block, 0, stream>>>(
          in_idx,
          num_in,
          static_cast<IdType>(comm_size-1), // bit mask
          proc_id_in);
      CUDA_CALL(cudaGetLastError());
    }
  }

  // then create a permutation array that groups processors together by
  // performing a radix sort
  IdType * proc_id_out = static_cast<IdType*>(
      device->AllocWorkspace(ctx, sizeof(IdType)*num_in));
  IdType * perm_out = static_cast<IdType*>(device->AllocWorkspace(ctx,
          sizeof(IdType)*num_in)); 
  {
    IdArray perm_in = aten::Range(0, num_in, sizeof(IdType)*8, ctx);

    size_t sort_workspace_size;
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(nullptr, sort_workspace_size,
        proc_id_in, proc_id_out, static_cast<IdType*>(perm_in->data), perm_out,
        num_in, 0, comm_bits, stream));

    void * sort_workspace = device->AllocWorkspace(ctx, sort_workspace_size);
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(sort_workspace, sort_workspace_size,
        proc_id_in, proc_id_out, static_cast<IdType*>(perm_in->data), perm_out,
        num_in, 0, comm_bits, stream));
    device->FreeWorkspace(ctx, sort_workspace);
  }
  device->FreeWorkspace(ctx, proc_id_out);
  device->FreeWorkspace(ctx, proc_id_in);

  // perform a histogram and then prefixsum on the sorted proc_id vector

  // finally, permute the input arrays
  // sort the data into continuous buffers for sending
  {
    const dim3 block(256);
    const dim3 grid((num_in+block.x-1)/block.x);

    _DualPermKernel<<<grid, block, 0, stream>>>(
        in_idx,
        in_value,
        perm_out,
        num_in,
        num_feat,
        out_idx,
        out_value);
    CUDA_CALL(cudaGetLastError());
  }

  // Count the number of values to be sent to each processor 
  {
    constexpr const int BLOCK_SIZE = 256;
    constexpr const int TILE_SIZE = 1024;
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_in+TILE_SIZE-1)/TILE_SIZE);

    if (comm_size < 128) {
      _CountIndexByRemainder<IdType, 128, BLOCK_SIZE, TILE_SIZE><<<
          grid, block, 0, stream>>>(
            out_idx,
            num_in,
            out_counts,
            comm_size);
      CUDA_CALL(cudaGetLastError());
    } else {
      CHECK_LE(comm_size, 1024) << "_CAPI_DGLNCCLSparseAllToAll() is not "
          "implemented for comms greater than 1024 ranks.";
      _CountIndexByRemainder<IdType, 1024, BLOCK_SIZE, TILE_SIZE><<<
          grid, block, 0, stream>>>(
            out_idx,
            num_in,
            out_counts,
            comm_size);
      CUDA_CALL(cudaGetLastError());
    }
  }
}

template<typename IdType, typename DType>
std::pair<IdArray, NDArray> SparseExchange(
    NCCLCommunicatorRef comm,
    IdArray in_idx,
    NDArray in_value,
    const int mode_id) {
  CHECK_EQ(in_idx->shape[0], in_value->shape[0]);

  const auto& ctx = in_idx->ctx;
  CHECK_EQ(ctx, in_value->ctx);
  auto device = DeviceAPI::Get(ctx);

  // TODO(dlasalle): Get the stream from the device context.
  cudaStream_t stream = 0;

  CHECK_EQ(in_idx->ndim, 1);

  const int64_t num_in = in_idx->shape[0];
  int64_t num_feat = 1;
  for (int d = 1; d < in_value->ndim; ++d) {
    num_feat *= in_value->shape[d];
  }

  const int64_t comm_size = comm->size();

  if (comm_size == 1) {
    // nothing to do, just return original arrays
    return std::pair<IdArray, NDArray>(in_idx, in_value);
  }

  IdType * send_idx = static_cast<IdType*>(device->AllocWorkspace(ctx,
      num_in*sizeof(IdType)));
  DType * send_value = static_cast<DType*>(device->AllocWorkspace(ctx,
      num_in*num_feat*sizeof(DType)));
  int64_t * send_sum = static_cast<int64_t*>(device->AllocWorkspace(ctx,
      (comm_size+1)*sizeof(int64_t)));

  CHECK_EQ(mode_id, static_cast<int>(AllToAllMode::REMAINDER));
  GenerateSparseBuffersFromRemainder(
      device,
      ctx,
      comm_size,
      num_in,
      num_feat,
      static_cast<const IdType*>(in_idx->data),
      static_cast<const DType*>(in_value->data),
      send_idx,
      send_value,
      send_sum,
      stream);

  // compute the prefix sum of the send values
  int64_t * send_prefix = static_cast<int64_t*>(
      device->AllocWorkspace(ctx, sizeof(int64_t)*(comm_size+1)));
  {
    size_t prefix_workspace_size;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_workspace_size,
        send_sum, send_prefix, comm_size+1));

    void * prefix_workspace = device->AllocWorkspace(
        ctx, prefix_workspace_size);
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(prefix_workspace, prefix_workspace_size,
        send_sum, send_prefix, comm_size+1));
    device->FreeWorkspace(ctx, prefix_workspace);
  }

  std::vector<int64_t> send_prefix_host(comm_size+1);
  device->CopyDataFromTo(
      send_prefix,
      0,
      send_prefix_host.data(),
      0,
      send_prefix_host.size()*sizeof(*send_prefix),
      ctx,
      DGLContext{kDLCPU, 0},
      DGLType{kDLInt, sizeof(*send_prefix)*8, 1},
      stream);
  device->FreeWorkspace(ctx, send_prefix);
  CHECK_EQ(send_prefix_host.back(), num_in);

  // communicate the amount to send
  int64_t * recv_sum = static_cast<int64_t*>(
      device->AllocWorkspace(ctx, sizeof(int64_t)*(comm_size+1)));
  comm->AllToAll(send_sum, recv_sum, 1, stream);

  // compute the prefix sum of the recv values
  int64_t * recv_prefix = static_cast<int64_t*>(
      device->AllocWorkspace(ctx, sizeof(int64_t)*(comm_size+1)));
  {
    size_t prefix_workspace_size;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_workspace_size,
        recv_sum, recv_prefix, comm_size+1));

    void * prefix_workspace = device->AllocWorkspace(
        ctx, prefix_workspace_size);
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(prefix_workspace, prefix_workspace_size,
        recv_sum, recv_prefix, comm_size+1));
    device->FreeWorkspace(ctx, prefix_workspace);
  }

  // finally copy the prefixsum sum down to the host
  std::vector<int64_t> recv_prefix_host(comm_size+1);
  device->CopyDataFromTo(
      recv_prefix,
      0,
      recv_prefix_host.data(),
      0,
      recv_prefix_host.size()*sizeof(*recv_prefix),
      ctx,
      DGLContext{kDLCPU, 0},
      DGLType{kDLInt, sizeof(*recv_prefix)*8, 1},
      stream);
  device->FreeWorkspace(ctx, recv_prefix);

  // use an event to track when copying is done
  cudaEvent_t d2h;
  cudaEventCreate(&d2h);
  cudaEventRecord(d2h, stream);

  // allocate output space
  cudaEventSynchronize(d2h);
  cudaEventDestroy(d2h);

  IdArray recv_idx = aten::NewIdArray(recv_prefix_host.back(), ctx, sizeof(IdType)*8);

  std::vector<int64_t> value_shape(in_value->ndim, 0);
  value_shape[0] = recv_prefix_host.back();
  for (int d = 1; d < in_value->ndim; ++d) {
    value_shape[d] = in_value->shape[d];
  }
  NDArray recv_value = NDArray::Empty(value_shape, in_value->dtype, ctx);

  // send data
  comm->SparseAllToAll(
      send_idx,
      send_value,
      num_feat,
      send_prefix_host.data(),
      static_cast<IdType*>(recv_idx->data),
      static_cast<DType*>(recv_value->data),
      recv_prefix_host.data(),
      stream);
  device->FreeWorkspace(ctx, send_idx);
  device->FreeWorkspace(ctx, send_value);

  return std::pair<IdArray, NDArray>(recv_idx, recv_value);
}

int NCCLCommunicator::size() const {
  return size_;
}

int NCCLCommunicator::rank() const {
  return rank_;
}

/* CAPI **********************************************************************/

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLGetUniqueId")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  *rv = NCCLUniqueIdRef(std::make_shared<NCCLUniqueId>());
});

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLCreateComm")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const int size = args[0];
  const int rank = args[1];
  NCCLUniqueIdRef idObj = args[2];

  *rv = NCCLCommunicatorRef(std::make_shared<NCCLCommunicator>(size, rank,
        idObj->Get()));
});

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLSparseAllToAll")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NCCLCommunicatorRef comm = args[0];
  IdArray in_idx = args[1];
  NDArray in_values = args[2];
  const int mode_id = args[3];

  List<ObjectRef> ret;
  ATEN_ID_TYPE_SWITCH(in_idx->dtype, IdType, {
    ATEN_DTYPE_SWITCH(in_values->dtype, DType, "values", {
      auto result = SparseExchange<IdType, DType>(comm, in_idx, in_values, mode_id);
      ret.push_back(Value(MakeValue(result.first)));
      ret.push_back(Value(MakeValue(result.second)));
    });
  });

  *rv = ret;
});

}
}
}

#endif
