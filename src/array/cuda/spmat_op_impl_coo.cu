/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/spmat_op_impl_csr.cu
 * \brief COO operator GPU implementation
 */

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include <vector>
#include <unordered_set>
#include <numeric>
#include <cstdint>
#include <cub/cub.cuh>
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"



namespace dgl {

using runtime::NDArray;
using runtime::DeviceAPI;

namespace aten {
namespace impl {


namespace
{

template<typename IdType>
struct EmptyKey {
  constexpr static const IdType value = static_cast<IdType>(-1);
};


inline __device__ int64_t atomicCAS(
    int64_t * const address,
    const int64_t compare,
    const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = unsigned long long int; // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return ::atomicCAS(reinterpret_cast<Type*>(address),
                   static_cast<Type>(compare),
                   static_cast<Type>(val));
}


inline __device__ int32_t atomicCAS(
    int32_t * const address,
    const int32_t compare,
    const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int; // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return ::atomicCAS(reinterpret_cast<Type*>(address),
                   static_cast<Type>(compare),
                   static_cast<Type>(val));
}




template<typename IdType>
inline __device__ bool attempt_insert_at(
    const size_t pos,
    const IdType id,
    IdType * const table) {
  const IdType key = atomicCAS(&table[pos], EmptyKey<IdType>::value, id);
  return key == EmptyKey<IdType>::value;
}


template<typename IdType>
inline __device__ size_t insert_hashmap(
    const IdType id,
    IdType * const table,
    const size_t table_size) {
  size_t pos = id % table_size;

  // linearly scan for an empty slot or matching entry
  IdType delta = 1;
  while (!attempt_insert_at(pos, id, table)) {
    pos = (pos+delta) % table_size;
    delta +=1;
  }

  return pos;
}


template<typename IdType>
inline __device__ IdType search_hashmap_for_pos(
    const IdType id,
    const IdType * const table,
    const IdType table_size) {
  IdType pos = id % table_size;

  // linearly scan for matching entry
  IdType delta = 1;
  while (table[pos] != id) {
    if (table[pos] == EmptyKey<IdType>::value) {
      return EmptyKey<IdType>::value;
    }

    pos = (pos+delta) % table_size;
    delta +=1;
  }
  assert(pos < table_size);

  return pos;
}


template<typename IdType>
inline __device__ const IdType * search_hashmap(
    const IdType id,
    const IdType * const table,
    const IdType table_size) {
  const IdType pos = search_hashmap_for_pos(id, table, table_size);

  if (pos != EmptyKey<IdType>::value) {
    return table+pos;
  } else {
    return nullptr;
  }
}


template<typename IdType>
inline __device__ bool hashmap_has(
    const IdType id,
    const IdType * const table,
    const IdType table_size) {
  return search_hashmap(id, table, table_size) != nullptr;
}


template<typename IdType, int BLOCK_SIZE>
__global__ void populate_hashmap(
  const IdType * rows,
  const int64_t num_rows,
  IdType * const hashmap,
  const size_t hashmap_size)
{
  int64_t row = threadIdx.x + blockIdx.x*BLOCK_SIZE;
  if (row < num_rows) {
    insert_hashmap(rows[row], hashmap, hashmap_size);
  }
}

template<typename IdType, int BLOCK_SIZE>
__global__ void count_edges(
  const IdType * const hashmap,
  const size_t hashmap_size,
  const IdType * const dsts,
  const int64_t num_edges,
  int64_t * const num_dsts_per_block)
{
  typedef cub::BlockReduce<uint16_t, BLOCK_SIZE> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  int64_t edge_index = threadIdx.x + blockIdx.x*BLOCK_SIZE;

  uint16_t flag = 0;
  if (edge_index < num_edges) {
    const IdType dst = dsts[edge_index];

    flag = hashmap_has<IdType>(dst, hashmap, hashmap_size);
  }

  flag = BlockReduce(temp_storage).Sum(flag);

  if (threadIdx.x == 0) {
    num_dsts_per_block[blockIdx.x] = flag;
  }
}


template<typename IdType, int BLOCK_SIZE>
__global__ void collect_edges(
  const IdType * const hashmap,
  const size_t hashmap_size,
  const IdType * const dsts,
  const IdType * const srcs,
  const IdType * const data,
  const int64_t num_edges,
  const int64_t * const prefix_dsts_per_block,
  IdType * const rows_out,
  IdType * const cols_out,
  IdType * const data_out)
{
  using BlockScan = typename cub::BlockScan<uint16_t, BLOCK_SIZE>;

  __shared__ typename BlockScan::TempStorage temp_storage;

  int64_t edge_index = threadIdx.x + blockIdx.x*BLOCK_SIZE;

  uint16_t flag = 0;
  if (edge_index < num_edges) {
    const IdType dst = dsts[edge_index];

    flag = hashmap_has<IdType>(dst, hashmap, hashmap_size);
  }

  uint16_t local_offset;
  BlockScan(temp_storage).ExclusiveSum(flag, local_offset);

  if (flag) {
    const int64_t out_index = local_offset+prefix_dsts_per_block[blockIdx.x];
    rows_out[out_index] = dsts[edge_index];
    cols_out[out_index] = srcs[edge_index];
    if (data) {
      assert(data_out);
      data_out[out_index] = data[edge_index];
    }
  }
}

}  // namespace

template <DLDeviceType XPU, typename IdType>
COOMatrix COOSliceRows(COOMatrix coo, NDArray rows) {
  // In the future, if the 'coo' is sorted and 'rows' was small, it may be
  // faster to do a binary search for rows. For a matrix with n rows, m
  // non-zeros, and we're slicing k rows, the complexity is O(k+m), where as
  // binary search will be O(k log m).

  constexpr const int BLOCK_SIZE = 128;

  DGLContext ctx = coo.row->ctx;
  auto device = DeviceAPI::Get(ctx); 

  // use default stream for now
  cudaStream_t stream = 0;

  const int row_grid_size = (rows->shape[0]+BLOCK_SIZE-1) / BLOCK_SIZE;
  const int edge_grid_size = (coo.row->shape[0]+BLOCK_SIZE-1) / BLOCK_SIZE;

  const size_t num_bits = static_cast<size_t>(
      std::ceil(std::log2(static_cast<double>(rows->shape[0]))));
  const size_t next_pow2 = 1ull << num_bits;
  // make sure how hash size is not a power of 2
  const size_t hash_size = (next_pow2 << 4)-1;

  // First we need create a hash map of rows we will keep, so for a given
  // edge we can check if it is gathered by this slice
  IdType * hash_table = static_cast<IdType*>(device->AllocWorkspace(ctx, hash_size*sizeof(IdType)));
  CUDA_CALL(cudaMemsetAsync(hash_table, EmptyKey<IdType>::value, hash_size*sizeof(IdType), stream)); 
  populate_hashmap<IdType, BLOCK_SIZE><<<row_grid_size, BLOCK_SIZE, 0, stream>>>(
      rows.Ptr<IdType>(), rows->shape[0], hash_table, hash_size);
  CUDA_CALL(cudaGetLastError());

  // Next we need to count the number of non-zeros that will be kept per
  // threadblock
  int64_t * block_prefix = static_cast<int64_t*>(device->AllocWorkspace(ctx, (edge_grid_size+1)*sizeof(int64_t)));
  count_edges<IdType, BLOCK_SIZE><<<
      edge_grid_size, BLOCK_SIZE, 0, stream>>>(
      hash_table, hash_size, coo.row.Ptr<IdType>(), coo.row->shape[0],
      block_prefix);
  CUDA_CALL(cudaGetLastError());

  // Generate a prefix sum of non-zeros per thread block, so that edges can be
  // inserted into the same array
  size_t prefix_sum_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr,
      prefix_sum_bytes,
      block_prefix,
      block_prefix,
      edge_grid_size+1));
  
  void * prefix_sum_space = device->AllocWorkspace(ctx, prefix_sum_bytes);

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      prefix_sum_space,
      prefix_sum_bytes,
      block_prefix,
      block_prefix,
      edge_grid_size+1));

  device->FreeWorkspace(ctx, prefix_sum_space);

  int64_t nnz;
  CUDA_CALL(cudaMemcpyAsync(
      &nnz, block_prefix+edge_grid_size, sizeof(nnz), cudaMemcpyDeviceToHost,
      stream));
  device->StreamSync(ctx, stream);
  
  // allocate output space
  IdArray ret_row = NDArray::Empty({nnz}, coo.row->dtype, ctx);
  IdArray ret_col = NDArray::Empty({nnz}, coo.row->dtype, ctx);
  IdArray ret_data = IsNullArray(coo.data) ? NullArray() : NDArray::Empty({nnz}, coo.row->dtype, ctx);

  // Finally insert the rows associated with the edges
  collect_edges<IdType, BLOCK_SIZE><<<
      edge_grid_size, BLOCK_SIZE, 0, stream>>>(
      hash_table, hash_size, coo.row.Ptr<IdType>(),
      coo.col.Ptr<IdType>(),
      IsNullArray(coo.data) ? nullptr : coo.data.Ptr<IdType>(),
      coo.row->shape[0],
      block_prefix, ret_row.Ptr<IdType>(), ret_col.Ptr<IdType>(),
      ret_data.Ptr<IdType>());
  CUDA_CALL(cudaGetLastError());

  device->FreeWorkspace(ctx, block_prefix);
  device->FreeWorkspace(ctx, hash_table);

  // Return the new matrix -- we preserve the order of the rows and columns
  return COOMatrix(rows->shape[0], coo.num_cols,
      ret_row, ret_col, ret_data, coo.row_sorted, coo.col_sorted);
}

template COOMatrix COOSliceRows<kDLGPU, int32_t>(COOMatrix , NDArray);
template COOMatrix COOSliceRows<kDLGPU, int64_t>(COOMatrix , NDArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
