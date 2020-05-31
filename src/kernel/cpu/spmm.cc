#include "./spmm.h"
#include <dgl/array.h>
#include "../binary_reduce.h"

namespace dgl {
namespace kernel {

template <int XPU, typename IdType, typename DType>
void SpMMCsr(const std::string& op, const std::string& reduce,
             const aten::CSRMatrix& csr,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  if (reduce == "sum") {
    SWITCH_OP(op, Op, {
      cpu::SpMMSumCsr<IdType, DType, Op>(csr, ufeat, efeat, out);
    });
  } else if (reduce == "max" || reduce == "min") {
    SWITCH_OP(op, Op, {
      if (reduce == "max")
        cpu::SpMMCmpCsr<IdType, DType, Op, cpu::op::Max<DType>>(
            csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
      else
        cpu::SpMMCmpCsr<IdType, DType, Op, cpu::op::Min<DType>>(
            csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else {
    LOG(FATAL) << "Unsupported SpMM reducer: " << reduce;
  }
}

template void SpMMCsr<kDLCPU, int32_t, float>(
    const std::string& op, const std::string& reduce, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLCPU, int64_t, float>(
    const std::string& op, const std::string& reduce, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLCPU, int32_t, double>(
    const std::string& op, const std::string& reduce, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLCPU, int64_t, double>(
    const std::string& op, const std::string& reduce, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);


template <int XPU, typename IdType, typename DType>
void SpMMBcastCsr(const std::string& op, const std::string& reduce,
                  const BcastInfo& info,
                  const aten::CSRMatrix& csr,
                  NDArray ufeat,
                  NDArray efeat,
                  NDArray out,
                  std::vector<NDArray> out_aux) {
  // TODO
  LOG(FATAL) << "Not implemented";
}

template void SpMMBcastCsr<kDLCPU, int32_t, float>(
    const std::string& op, const std::string& reduce,
    const BcastInfo& info,
    const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMBcastCsr<kDLCPU, int64_t, float>(
    const std::string& op, const std::string& reduce,
    const BcastInfo& info,
    const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMBcastCsr<kDLCPU, int32_t, double>(
    const std::string& op, const std::string& reduce,
    const BcastInfo& info,
    const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMBcastCsr<kDLCPU, int64_t, double>(
    const std::string& op, const std::string& reduce,
    const BcastInfo& info,
    const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);


template <int XPU, typename IdType, typename DType>
void SpMMCoo(const std::string& op, const std::string& reduce,
             const aten::COOMatrix& coo,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  // TODO
  LOG(FATAL) << "Not implemented";
}

template void SpMMCoo<kDLCPU, int32_t, float>(
    const std::string& op, const std::string& reduce, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLCPU, int64_t, float>(
    const std::string& op, const std::string& reduce, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLCPU, int32_t, double>(
    const std::string& op, const std::string& reduce, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLCPU, int64_t, double>(
    const std::string& op, const std::string& reduce, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

template <int XPU, typename IdType, typename DType>
void SpMMBcastCoo(const std::string& op, const std::string& reduce,
                  const BcastInfo& info,
                  const aten::COOMatrix& coo,
                  NDArray ufeat,
                  NDArray efeat,
                  NDArray out,
                  std::vector<NDArray> out_aux) {
  // TODO
  LOG(FATAL) << "Not implemented";
}

template void SpMMBcastCoo<kDLCPU, int32_t, float>(
    const std::string& op, const std::string& reduce,
    const BcastInfo& info,
    const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMBcastCoo<kDLCPU, int64_t, float>(
    const std::string& op, const std::string& reduce,
    const BcastInfo& info,
    const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMBcastCoo<kDLCPU, int32_t, double>(
    const std::string& op, const std::string& reduce,
    const BcastInfo& info,
    const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMBcastCoo<kDLCPU, int64_t, double>(
    const std::string& op, const std::string& reduce,
    const BcastInfo& info,
    const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

}  // namespace kernel
}  // namespace dgl