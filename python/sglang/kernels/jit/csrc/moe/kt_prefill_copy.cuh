// Fused row copies for KT prefill staging.
//
// The resident side copies the same logical expert rows for every MXFP4 raw
// bank (w13 weight/scale, w2 weight/scale), while each bank has a different
// row size.  The host->device side already batches a whole pinned generation;
// using one cudaMemcpyBatchAsync there removes the per-run runtime calls.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sglang::kt {

namespace {

constexpr int kCopyThreads = 256;
constexpr int kCopyBlocksPerBank = 64;

struct MultiRowCopyParams {
  const int64_t* __restrict__ dst_ptrs;
  const int64_t* __restrict__ src_ptrs;
  const int64_t* __restrict__ feat_bytes;
  const int32_t* __restrict__ swap_halves;
  const void* __restrict__ dst_indices;
  const void* __restrict__ src_indices;
  int64_t length;
  int num_banks;
};

template <typename IdType, int kThreads, int kBlocksPerBank>
__global__ __launch_bounds__(kThreads) void multi_row_copy_kernel(
    const MultiRowCopyParams __grid_constant__ params) {
  const int bank = static_cast<int>(blockIdx.x / kBlocksPerBank);
  if (bank >= params.num_banks) {
    return;
  }

  const int block_in_bank = static_cast<int>(blockIdx.x % kBlocksPerBank);
  const int64_t feat = params.feat_bytes[bank];
  if (feat <= 0 || (feat & 15) != 0) {
    return;
  }

  const bool swap_halves = params.swap_halves && params.swap_halves[bank] != 0;
  if (swap_halves && (feat & 31) != 0) {
    return;
  }

  const auto* src = reinterpret_cast<const uint8_t*>(params.src_ptrs[bank]);
  auto* dst = reinterpret_cast<uint8_t*>(params.dst_ptrs[bank]);
  const auto* dst_indices = static_cast<const IdType*>(params.dst_indices);
  const auto* src_indices = static_cast<const IdType*>(params.src_indices);
  const int64_t units_per_row = feat >> 4;
  const int64_t total_units = params.length * units_per_row;
  const int64_t stride =
      static_cast<int64_t>(kBlocksPerBank) * static_cast<int64_t>(kThreads);
  const int64_t half = feat >> 1;

  for (int64_t unit =
           static_cast<int64_t>(block_in_bank) * kThreads + threadIdx.x;
       unit < total_units;
       unit += stride) {
    const int64_t row = unit / units_per_row;
    const int64_t dst_col = (unit - row * units_per_row) << 4;
    int64_t src_col = dst_col;
    if (swap_halves) {
      src_col = dst_col < half ? dst_col + half : dst_col - half;
    }
    const int64_t dst_row = static_cast<int64_t>(dst_indices[row]);
    const int64_t src_row = static_cast<int64_t>(src_indices[row]);
    const uint4 value = *reinterpret_cast<const uint4*>(
        src + src_row * feat + src_col);
    *reinterpret_cast<uint4*>(dst + dst_row * feat + dst_col) = value;
  }
}

}  // namespace

struct MultiRowCopy {
  static void run(
      const tvm::ffi::TensorView dst_ptrs,
      const tvm::ffi::TensorView src_ptrs,
      const tvm::ffi::TensorView feat_bytes,
      const tvm::ffi::TensorView swap_halves,
      const tvm::ffi::TensorView dst_indices,
      const tvm::ffi::TensorView src_indices) {
    using namespace host;

    auto num_banks = SymbolicSize{"num_banks"};
    auto length = SymbolicSize{"row_count"};
    auto indices_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({num_banks})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(dst_ptrs)
        .verify(src_ptrs)
        .verify(feat_bytes);
    TensorMatcher({num_banks})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(swap_halves);
    TensorMatcher({length})
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device(device)
        .verify(dst_indices)
        .verify(src_indices);

    const int64_t banks = num_banks.unwrap();
    const int64_t rows = length.unwrap();
    RuntimeCheck(banks > 0, "KT multi-row copy requires at least one bank");
    if (rows == 0) {
      return;
    }

    const auto params = MultiRowCopyParams{
        .dst_ptrs = static_cast<const int64_t*>(dst_ptrs.data_ptr()),
        .src_ptrs = static_cast<const int64_t*>(src_ptrs.data_ptr()),
        .feat_bytes = static_cast<const int64_t*>(feat_bytes.data_ptr()),
        .swap_halves = static_cast<const int32_t*>(swap_halves.data_ptr()),
        .dst_indices = dst_indices.data_ptr(),
        .src_indices = src_indices.data_ptr(),
        .length = rows,
        .num_banks = static_cast<int>(banks),
    };

    const bool use_int32 = indices_dtype.unwrap().bits == 32;
    const auto kernel = use_int32
        ? multi_row_copy_kernel<int32_t, kCopyThreads, kCopyBlocksPerBank>
        : multi_row_copy_kernel<int64_t, kCopyThreads, kCopyBlocksPerBank>;
    LaunchKernel(
        static_cast<uint32_t>(banks * kCopyBlocksPerBank),
        kCopyThreads,
        device.unwrap())(kernel, params);
  }
};

struct BatchMemcpy {
  static void run(
      const tvm::ffi::TensorView dst_ptrs,
      const tvm::ffi::TensorView src_ptrs,
      const tvm::ffi::TensorView sizes,
      int64_t stream_handle) {
    using namespace host;

    auto count = SymbolicSize{"batch_count"};
    TensorMatcher({count})
        .with_dtype<int64_t>()
        .with_device<kDLCPU>()
        .verify(dst_ptrs)
        .verify(src_ptrs)
        .verify(sizes);

    const int64_t n = count.unwrap();
    if (n == 0) {
      return;
    }

#if CUDART_VERSION >= 13000
    RuntimeCheck(
        stream_handle != 0,
        "cudaMemcpyBatchAsync rejects the legacy NULL stream");
    auto attributes = ::cudaMemcpyAttributes{};
    attributes.srcAccessOrder = ::cudaMemcpySrcAccessOrderStream;
    std::size_t attribute_index = 0;
    const cudaError_t status = ::cudaMemcpyBatchAsync(
        reinterpret_cast<void* const*>(dst_ptrs.data_ptr()),
        reinterpret_cast<const void* const*>(src_ptrs.data_ptr()),
        reinterpret_cast<const std::size_t*>(sizes.data_ptr()),
        static_cast<std::size_t>(n),
        &attributes,
        &attribute_index,
        1,
        reinterpret_cast<::cudaStream_t>(stream_handle));
    RuntimeCheck(
        status == cudaSuccess,
        "cudaMemcpyBatchAsync failed: ",
        ::cudaGetErrorString(status));
#else
    RuntimeCheck(
        false,
        "KT batch H2D requires CUDA >= 13.0 (compiled with CUDART_VERSION=",
        CUDART_VERSION,
        ")");
#endif
  }
};

}  // namespace sglang::kt
