# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.model_executor.layers.fused_moe.oracle import int_wna16
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    _backend_incompatibility_reason,
    _convert_moe_wna16_humming_tensors,
    convert_to_wna16_moe_kernel_format,
    map_wna16_backend,
)
from vllm.model_executor.layers.quantization import moe_wna16
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config,
    MoeWNA16Method,
)


def test_map_wna16_backend_supports_triton():
    assert map_wna16_backend("triton") == WNA16MoEBackend.TRITON


@pytest.mark.parametrize(
    (
        "backend",
        "quant_overrides",
        "input_dtype",
        "has_zero_points",
        "has_bias",
        "eligible",
    ),
    [
        (WNA16MoEBackend.MARLIN, {}, None, False, False, True),
        (WNA16MoEBackend.BATCHED_MARLIN, {}, None, False, False, False),
        (WNA16MoEBackend.MARLIN, {"num_bits": 8}, None, False, False, False),
        (WNA16MoEBackend.MARLIN, {"symmetric": False}, None, False, False, False),
        (WNA16MoEBackend.MARLIN, {"group_size": 32}, None, False, False, False),
        (WNA16MoEBackend.MARLIN, {}, torch.int8, False, False, False),
        (
            WNA16MoEBackend.MARLIN,
            {"actorder": ActivationOrdering.GROUP},
            None,
            False,
            False,
            False,
        ),
        (WNA16MoEBackend.MARLIN, {}, None, True, False, False),
        (WNA16MoEBackend.MARLIN, {}, None, False, True, False),
    ],
)
def test_chunked_marlin_moe_repack_eligibility_is_narrow(
    backend,
    quant_overrides,
    input_dtype,
    has_zero_points,
    has_bias,
    eligible,
):
    quant_kwargs = {
        "num_bits": 4,
        "type": QuantizationType.INT,
        "strategy": QuantizationStrategy.GROUP,
        "symmetric": True,
        "dynamic": False,
        "group_size": 128,
    }
    quant_kwargs.update(quant_overrides)
    quant_config = QuantizationArgs(**quant_kwargs)
    optional_tensor = torch.empty(0)

    reason = int_wna16._compressed_tensors_chunking_reason(
        backend,
        quant_config,
        input_dtype,
        optional_tensor if has_zero_points else None,
        None,
        optional_tensor if has_bias else None,
        None,
    )

    assert (reason is None) is eligible


def test_chunked_marlin_moe_repack_synchronizes_on_error(monkeypatch):
    events = []
    monkeypatch.setattr(
        torch.accelerator,
        "device_index",
        lambda device_index: nullcontext(),
    )
    monkeypatch.setattr(
        torch.accelerator,
        "synchronize",
        lambda device=None: events.append("synchronize"),
    )
    monkeypatch.setattr(
        int_wna16,
        "empty_accelerator_view_from_host",
        lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )

    def fail_repack(*args, **kwargs):
        events.append("repack")
        raise RuntimeError("repack failed")

    monkeypatch.setattr(
        int_wna16.ops,
        "gptq_marlin_moe_repack_into",
        fail_repack,
    )

    with pytest.raises(RuntimeError, match="repack failed"):
        int_wna16._gptq_marlin_moe_repack_uva(
            torch.zeros((2, 2, 3), dtype=torch.int32),
            torch.empty((2, 0), dtype=torch.int32),
            size_k=32,
            size_n=3,
            num_bits=4,
            chunk_size=1,
            is_a_8bit=False,
        )

    assert events == ["repack", "synchronize"]


def test_compressed_tensors_process_preserves_chunked_uva_weights(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe_wna16_marlin as ct_marlin,
    )

    method = object.__new__(ct_marlin.CompressedTensorsWNA16MarlinMoEMethod)
    method.wna16_backend = WNA16MoEBackend.MARLIN
    method.weight_quant = object()
    method.marlin_input_dtype = None
    method.symmetric = True
    method.is_marlin = True
    method.experts_cls = object
    method.is_k_full = True
    method.moe = object()
    method.get_fused_moe_quant_config = lambda layer: object()

    layer = torch.nn.Module()
    parameter_names = (
        "w13_weight_packed",
        "w2_weight_packed",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_weight_g_idx",
        "w2_weight_g_idx",
        "w13_g_idx_sort_indices",
        "w2_g_idx_sort_indices",
    )
    for name in parameter_names:
        layer.register_parameter(
            name,
            torch.nn.Parameter(torch.zeros(2), requires_grad=False),
        )
    layer._expert_routing_tables = lambda: (None, None, None)

    w13 = torch.ones(2)
    w2 = torch.full((2,), 2.0)
    w13._vllm_is_uva_offloaded = True
    w2._vllm_is_uva_offloaded = True
    converted = (
        w13,
        w2,
        torch.ones(2),
        torch.ones(2),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    monkeypatch.setattr(
        ct_marlin,
        "convert_to_wna16_moe_kernel_format",
        lambda **kwargs: converted,
    )
    monkeypatch.setattr(
        ct_marlin,
        "make_wna16_moe_kernel",
        lambda **kwargs: object(),
    )

    method.process_weights_after_loading(layer)

    assert layer.w13_weight_packed._vllm_is_uva_offloaded
    assert layer.w2_weight_packed._vllm_is_uva_offloaded
    assert layer.w13_weight is layer.w13_weight_packed
    assert layer.w2_weight is layer.w2_weight_packed


@pytest.mark.parametrize(
    ("backend", "quant_config", "may_have_zp", "may_have_bias", "expected"),
    [
        (
            WNA16MoEBackend.TRITON,
            AutoAWQConfig(4, 128, True, False),
            True,
            False,
            "AutoAWQ weight layout",
        ),
        (
            WNA16MoEBackend.TRITON,
            AutoGPTQConfig(4, 128, True, True, False, {}, {}),
            False,
            False,
            "activation ordering",
        ),
        (
            WNA16MoEBackend.TRITON,
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.GROUP,
            ),
            False,
            False,
            "activation ordering",
        ),
        (
            WNA16MoEBackend.TRITON,
            AutoGPTQConfig(4, 128, False, True, False, {}, {}),
            False,
            True,
            "bias",
        ),
        (
            WNA16MoEBackend.MARLIN,
            MoeWNA16Config(
                linear_quant_method="gptq",
                weight_bits=4,
                group_size=128,
                has_zp=False,
                lm_head_quantized=False,
                modules_to_not_convert=None,
                full_config={},
            ),
            False,
            False,
            "MoeWNA16 checkpoint layout",
        ),
    ],
)
def test_wna16_oracle_rejects_incompatible_quant_structures(
    backend, quant_config, may_have_zp, may_have_bias, expected
):
    reason = _backend_incompatibility_reason(
        backend=backend,
        quant_config=quant_config,
        may_have_zp=may_have_zp,
        may_have_bias=may_have_bias,
    )

    assert reason is not None
    assert expected in reason


def test_compressed_tensors_weights_are_transposed_for_triton():
    quant_config = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=32,
    )
    w13 = torch.arange(16, dtype=torch.int32).reshape(1, 2, 8)
    w2 = torch.arange(12, dtype=torch.int32).reshape(1, 2, 6)
    w13_scale = torch.arange(32, dtype=torch.float16).reshape(1, 4, 8)
    w2_scale = torch.arange(18, dtype=torch.float16).reshape(1, 3, 6)

    converted = convert_to_wna16_moe_kernel_format(
        backend=WNA16MoEBackend.TRITON,
        layer=torch.nn.Module(),
        quant_config=quant_config,
        input_dtype=None,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
    )

    assert converted is not None
    assert torch.equal(converted[0], w13.transpose(1, 2).contiguous().view(torch.uint8))
    assert torch.equal(converted[1], w2.transpose(1, 2).contiguous().view(torch.uint8))
    assert torch.equal(converted[2], w13_scale.transpose(1, 2).contiguous())
    assert torch.equal(converted[3], w2_scale.transpose(1, 2).contiguous())


def test_moe_wna16_setup_forwards_selected_backend(monkeypatch):
    method = object.__new__(MoeWNA16Method)
    method.experts_cls = object
    method.wna16_backend = WNA16MoEBackend.HUMMING
    method.moe = object()
    quant_config = object()
    method.get_fused_moe_quant_config = lambda layer: quant_config
    layer = SimpleNamespace(_expert_routing_tables=lambda: (None, None, None))
    captured = {}
    kernel = object()

    def fake_make_wna16_moe_kernel(**kwargs):
        captured.update(kwargs)
        return kernel

    monkeypatch.setattr(moe_wna16, "make_wna16_moe_kernel", fake_make_wna16_moe_kernel)

    method._setup_kernel(layer)

    assert method.moe_kernel is kernel
    assert captured["backend"] == WNA16MoEBackend.HUMMING
    assert captured["layer"] is layer


def test_moe_wna16_humming_adapter_repacks_uint8_tensors():
    qweight = torch.arange(32, dtype=torch.uint8).reshape(1, 4, 8)
    scales = torch.arange(16, dtype=torch.float16).reshape(1, 4, 4)
    qzeros = torch.arange(16, dtype=torch.uint8).reshape(1, 8, 2)

    converted = _convert_moe_wna16_humming_tensors(
        {"qweight": qweight, "scales": scales, "qzeros": qzeros},
        has_zero_point=True,
    )

    assert torch.equal(converted["weight"], qweight.view(torch.int32))
    assert converted["weight"].shape == (1, 4, 2)
    assert torch.equal(converted["weight_scale"], scales)
    expected_qzeros = (
        qzeros.transpose(-1, -2)
        .contiguous()
        .view(torch.int32)
        .transpose(-1, -2)
        .contiguous()
    )
    assert torch.equal(converted["zero_point"], expected_qzeros)
    assert converted["zero_point"].shape == (1, 2, 2)


def test_moe_wna16_uses_humming_quant_config(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import humming_utils

    method = object.__new__(MoeWNA16Method)
    method.wna16_backend = WNA16MoEBackend.HUMMING
    layer = object()
    quant_config = object()
    monkeypatch.setattr(
        humming_utils,
        "get_humming_moe_quant_config",
        lambda actual_layer: quant_config if actual_layer is layer else None,
    )

    assert method.get_fused_moe_quant_config(layer) is quant_config
