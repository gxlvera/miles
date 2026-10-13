"""Make Megatron-Bridge's Qwen2.5-VL model compute what transformers and SGLang compute.

Installed in every training process through ``--custom-megatron-init-path``.

- Miles calls the model with ``fp32_output``. Qwen3VLModel.forward absorbs it through ``**kwargs``;
  Qwen25VLModel.forward has no ``**kwargs`` and raises. Drop it here, so both VL models keep their
  logits in the model dtype.
- The vision tower is transformers' module, and Megatron's Float16Module casts its buffers to bf16,
  ``inv_freq`` of the vision rotary embedding included. The angles then come out in bf16, off by up
  to half a radian for large images, and training no longer matches the rollout engine. Compute them
  in fp32 from the embedding's ``dim`` and ``theta``.

Packed (THD) mRoPE positions need no fix: Qwen25VLModel builds them per packed row itself.
"""

from __future__ import annotations

import importlib

import torch

_PATCHED = "_miles_qwen2_5_vl_patched"


def install(_args) -> None:
    try:
        model_mod = importlib.import_module("megatron.bridge.models.qwen_vl.modeling_qwen25_vl")
        hf_mod = importlib.import_module("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
    except ImportError:
        return
    _patch_forward_kwargs(model_mod.Qwen25VLModel)
    _patch_vision_rotary(hf_mod.Qwen2_5_VisionRotaryEmbedding)


def _patch_forward_kwargs(model_cls) -> None:
    if model_cls.__dict__.get(_PATCHED, False):
        return
    orig_forward = model_cls.forward

    def forward(self, *args, fp32_output=None, **kwargs):
        return orig_forward(self, *args, **kwargs)

    model_cls.forward = forward
    setattr(model_cls, _PATCHED, True)


def _patch_vision_rotary(rotary_cls) -> None:
    if rotary_cls.__dict__.get(_PATCHED, False):
        return

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(0, self.dim, 2, dtype=torch.float32, device=position_ids.device)
        inv_freq = 1.0 / (self.theta ** (steps / self.dim))
        return (position_ids.unsqueeze(-1).float() * inv_freq).flatten(1)

    rotary_cls.forward = forward
    setattr(rotary_cls, _PATCHED, True)
