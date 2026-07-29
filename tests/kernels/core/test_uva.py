# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc

import pytest
import torch

from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import (
    empty_accelerator_view_from_host,
    get_accelerator_view_from_cpu_tensor,
)

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_cpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10, 10, device="cpu", pin_memory=True, dtype=torch.int32)
    cuda_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert cuda_view.device.type == "cuda"

    assert cuda_view[0, 0] == 0
    assert cuda_view[2, 3] == 0
    assert cuda_view[4, 5] == 0

    cpu_tensor[0, 0] = 1
    cpu_tensor[2, 3] = 2
    cpu_tensor[4, 5] = -1

    cuda_view.mul_(2)
    assert cuda_view[0, 0] == 2
    assert cuda_view[2, 3] == 4
    assert cuda_view[4, 5] == -2


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_gpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10, 10, device="cpu", pin_memory=True, dtype=torch.int32)
    cuda_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert cuda_view.device.type == "cuda"

    assert cuda_view[0, 0] == 0
    assert cuda_view[2, 3] == 0
    assert cuda_view[4, 5] == 0

    cuda_view[0, 0] = 1
    cuda_view[2, 3] = 2
    cuda_view[4, 5] = -1
    cuda_view.mul_(2)

    assert cpu_tensor[0, 0] == 2
    assert cpu_tensor[2, 3] == 4
    assert cpu_tensor[4, 5] == -2


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_empty_host_view_lifetime(device):
    torch.accelerator.set_device_index(torch.device(device).index)
    torch.set_default_device(device)
    cuda_view = empty_accelerator_view_from_host((4, 6), torch.int32)
    assert cuda_view.device == torch.device(device)
    assert cuda_view.shape == (4, 6)
    assert cuda_view.dtype == torch.int32
    assert cuda_view.is_contiguous()

    cuda_view.copy_(torch.arange(24, dtype=torch.int32, device=device).view(4, 6))
    retained_view = cuda_view.view(-1)
    del cuda_view
    gc.collect()

    retained_view.add_(1)
    torch.accelerator.synchronize()
    expected = torch.arange(1, 25, dtype=torch.int32, device="cpu")
    torch.testing.assert_close(retained_view.cpu(), expected)
