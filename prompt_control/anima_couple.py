# Adapted from https://github.com/pamparamm/ComfyUI-ppm
from collections.abc import Callable
from functools import partial
from math import lcm

import torch
import torch.nn.functional as F
from comfy.ldm.anima.model import Anima as AnimaDIT
from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention
from comfy.patcher_extension import WrapperExecutor
from comfy.sampler_helpers import convert_cond
from comfy.samplers import process_conds


def reshape_mask(mask: torch.Tensor, size: tuple[int, int], bs: int, num_tokens: int) -> torch.Tensor:
    num_conds = mask.shape[0]

    mask_downsample = F.interpolate(mask, size=size, mode="nearest")
    mask_downsample_reshaped = mask_downsample.view(num_conds, num_tokens, 1).repeat_interleave(bs, dim=0)

    return mask_downsample_reshaped


def wrap_forwards(anima_model):
    backups = {}
    for block_name, b in (
        (n, b) for n, b in anima_model.named_modules() if "cross_attn" in n and isinstance(b, CosmosAttention)
    ):
        backups[block_name] = b.forward
        b.forward = partial(cosmos_attention_forward_couple, b.forward)
    return backups


def unwrap_forwards(anima_model, backups):
    for block_name, b in (
        (n, b) for n, b in anima_model.named_modules() if "cross_attn" in n and isinstance(b, CosmosAttention)
    ):
        b.forward = backups[block_name]


def anima_sample_wrapper(executor, *args, **kwargs):
    guider, _, extra_options, _, noise, latent_image, denoise_mask, *_ = args
    seed = extra_options["seed"]
    device = "cuda"  # TODO: fix

    def pc_process_conds(pc_conds):
        conds = [convert_cond([c])[0] for c in pc_conds]
        conds = process_conds(
            guider.inner_model,
            noise,
            {"positive": conds},
            device,
            latent_image,
            denoise_mask,
            seed,
            latent_shapes=[latent_image.shape],
        )
        return [
            c["model_conds"]["c_crossattn"].cond * pc_conds[i][1].get("strength", 1.0)
            for i, c in enumerate(conds["positive"])
        ]

    extra_options["model_options"]["transformer_options"]["pc_process_conds"] = pc_process_conds
    return executor(*args, **kwargs)


def anima_forward_wrapper(executor: WrapperExecutor, *args, **kwargs):
    """Model wrapper does something with activation shapes?"""
    anima_model: AnimaDIT = executor.class_obj  # type: ignore

    x: torch.Tensor = args[0]
    transformer_options: dict = kwargs.get("transformer_options", {}).copy()
    pc = transformer_options.get("pc_couple")
    if pc and "processed_conds" not in pc:
        pc["processed_conds"] = transformer_options["pc_process_conds"](pc["conds"])
        pc["mask"] = pc["mask"].to(x.device)
    patch_spatial = anima_model.patch_spatial

    activations_shape = list(x.shape)
    activations_shape[-2] = activations_shape[-2] // patch_spatial
    activations_shape[-1] = activations_shape[-1] // patch_spatial

    transformer_options["activations_shape"] = activations_shape
    kwargs["transformer_options"] = transformer_options

    b = {}
    if pc:
        b = wrap_forwards(anima_model)
    r = executor(*args, **kwargs)
    if pc:
        unwrap_forwards(anima_model, b)
    return r


def cosmos_attention_forward_couple(_forward: Callable, x, context, rope_emb, transformer_options):
    """attention block wrapper"""
    if "pc_couple" not in transformer_options:
        return _forward(x, context, rope_emb, transformer_options)
    c: torch.Tensor = context

    args = transformer_options["pc_couple"]

    mask = args["mask"]
    base_strength = args["base_strength"]
    conds = args["processed_conds"][1:]
    num_conds = len(conds) + 1
    num_tokens_c: list[int] = [cond.shape[1] for cond in conds]

    num_chunks = len(transformer_options["cond_or_uncond"])
    bs = x.shape[0] // num_chunks

    x_chunks = x.chunk(num_chunks, dim=0)
    c_chunks = c.chunk(num_chunks, dim=0)
    lcm_tokens_c = lcm(c.shape[1], *num_tokens_c)
    conds_c_tensor = torch.cat(
        [cond.repeat(bs, lcm_tokens_c // num_tokens_c[i], 1) for i, cond in enumerate(conds)],
        dim=0,
    )

    xs, cs = [], []
    for i in range(num_chunks):
        c_target = c_chunks[i].repeat(1, lcm_tokens_c // c.shape[1], 1) * base_strength
        xs.append(x_chunks[i].repeat(num_conds, 1, 1))
        cs.append(torch.cat([c_target, conds_c_tensor], dim=0))

    out = _forward(torch.cat(xs, dim=0), torch.cat(cs, dim=0), rope_emb, transformer_options)

    size = tuple(transformer_options["activations_shape"][-2:])
    mask_downsample = reshape_mask(mask, size, bs, out.shape[1])

    rows = num_conds * bs
    outputs = []
    for i in range(num_chunks):
        chunk = out[i * rows : (i + 1) * rows] * mask_downsample
        outputs.append(chunk.view(num_conds, bs, *chunk.shape[1:]).sum(0))

    return torch.cat(outputs, dim=0)
