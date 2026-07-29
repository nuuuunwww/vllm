#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/version.h>
#include <cuda_runtime.h>

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace {

struct CudaHostDeleter {
  void operator()(void* ptr) const {
    if (ptr != nullptr) {
      cudaFreeHost(ptr);
    }
  }
};

}  // namespace

// This function assumes that `cpu_tensor` is a CPU tensor,
// and that UVA (Unified Virtual Addressing) is enabled.
torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor) {
  STD_TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  const auto dtype = cpu_tensor.scalar_type();
  const auto layout = cpu_tensor.layout();
  const torch::stable::Device cuda_dev(torch::headeronly::DeviceType::CUDA);

  // handle empty tensor
  if (cpu_tensor.numel() == 0) {
    return torch::stable::empty(cpu_tensor.sizes(), dtype, layout, cuda_dev);
  }

  std::array<StableIValue, 2> is_pinned_stack{
      torch::stable::detail::from(cpu_tensor),
      torch::stable::detail::from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::is_pinned", "", is_pinned_stack.data(), TORCH_ABI_VERSION));
  if (torch::stable::detail::to<bool>(is_pinned_stack[0])) {
    // If CPU tensor is pinned, directly get the device pointer.
    void* host_ptr = const_cast<void*>(cpu_tensor.mutable_data_ptr());
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    STD_TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer failed: ",
                    cudaGetErrorString(err));

    return torch::stable::from_blob(
        device_ptr, cpu_tensor.sizes(), cpu_tensor.strides(), cuda_dev, dtype,
        [base = cpu_tensor](void*) {});  // keep cpu tensor alive
  }

  // If CPU tensor is not pinned, allocate a new pinned memory buffer.
  torch::stable::Tensor contiguous_cpu = torch::stable::contiguous(cpu_tensor);
  size_t nbytes = contiguous_cpu.numel() * contiguous_cpu.element_size();

  void* host_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&host_ptr, nbytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaHostAlloc failed: ", cudaGetErrorString(err));
  }

  err = cudaMemcpy(host_ptr, contiguous_cpu.const_data_ptr(), nbytes,
                   cudaMemcpyDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    STD_TORCH_CHECK(false, "cudaMemcpy failed: ", cudaGetErrorString(err));
  }

  void* device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    STD_TORCH_CHECK(
        false, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  }

  auto deleter = [host_ptr](void*) { cudaFreeHost(host_ptr); };

  return torch::stable::from_blob(device_ptr, contiguous_cpu.sizes(),
                                  contiguous_cpu.strides(), cuda_dev,
                                  contiguous_cpu.scalar_type(), deleter);
}

torch::stable::Tensor empty_cuda_view_from_host(
    torch::stable::Tensor& dtype_template, const std::vector<int64_t>& sizes) {
  STD_TORCH_CHECK(dtype_template.device().is_cpu(),
                  "dtype_template must be on CPU");
  STD_TORCH_CHECK(dtype_template.numel() == 0,
                  "dtype_template must have zero elements");

  const auto dtype = dtype_template.scalar_type();
  const auto layout = dtype_template.layout();
  int current_device = -1;
  cudaError_t err = cudaGetDevice(&current_device);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "cudaGetDevice failed: ", cudaGetErrorString(err));
  const torch::stable::Device cuda_dev(
      torch::headeronly::DeviceType::CUDA,
      static_cast<torch::stable::DeviceIndex>(current_device));

  int64_t numel = 1;
  for (const int64_t size : sizes) {
    STD_TORCH_CHECK(size >= 0, "Tensor dimensions must be non-negative");
    if (numel == 0 || size == 0) {
      numel = 0;
      continue;
    }
    STD_TORCH_CHECK(numel <= std::numeric_limits<int64_t>::max() / size,
                    "Tensor size overflow");
    numel *= size;
  }

  if (numel == 0) {
    return torch::stable::empty(sizes, dtype, layout, cuda_dev);
  }

  const size_t element_size = dtype_template.element_size();
  STD_TORCH_CHECK(static_cast<uint64_t>(numel) <=
                      std::numeric_limits<size_t>::max() / element_size,
                  "Tensor byte size overflow");
  const size_t nbytes = static_cast<size_t>(numel) * element_size;

  void* host_ptr = nullptr;
  err = cudaHostAlloc(&host_ptr, nbytes, cudaHostAllocMapped);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "cudaHostAlloc failed: ", cudaGetErrorString(err));
  std::unique_ptr<void, CudaHostDeleter> allocation(host_ptr);

  void* device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));

  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for (size_t i = sizes.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= sizes[i - 1];
  }

  auto deleter = [host_ptr](void*) { cudaFreeHost(host_ptr); };
  auto result = torch::stable::from_blob(device_ptr, sizes, strides, cuda_dev,
                                         dtype, deleter);
  allocation.release();
  return result;
}
