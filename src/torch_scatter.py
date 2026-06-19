"""Minimal torch_scatter compatibility shim for FRIGID smoke runs."""

from __future__ import annotations

import torch


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(f"dim {dim} out of range for tensor with {ndim} dims")
    return dim


def _expand_index(index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    if index.shape == src.shape:
        return index.long()
    return index.long().expand_as(src)


def _infer_dim_size(index: torch.Tensor, dim: int) -> int:
    if index.numel() == 0:
        return 0
    return int(index.max().item()) + 1


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: int | None = None) -> torch.Tensor:
    dim = _normalize_dim(dim, src.dim())
    index = _expand_index(index, src)
    if dim_size is None:
        dim_size = _infer_dim_size(index, dim)
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    out.scatter_add_(dim, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: int | None = None) -> torch.Tensor:
    summed = scatter_add(src, index, dim=dim, dim_size=dim_size)
    ones = torch.ones_like(src, dtype=summed.dtype)
    counts = scatter_add(ones, index, dim=dim, dim_size=dim_size)
    return summed / counts.clamp_min(1)


def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int | None = None,
):
    dim = _normalize_dim(dim, src.dim())
    index = _expand_index(index, src)
    if dim_size is None:
        dim_size = _infer_dim_size(index, dim)
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    if src.dtype.is_floating_point:
        out = torch.full(out_shape, -torch.inf, dtype=src.dtype, device=src.device)
    else:
        out = torch.full(out_shape, torch.iinfo(src.dtype).min, dtype=src.dtype, device=src.device)
    out.scatter_reduce_(dim, index, src, reduce="amax", include_self=True)
    return out, None


def scatter_min(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int | None = None,
):
    dim = _normalize_dim(dim, src.dim())
    index = _expand_index(index, src)
    if dim_size is None:
        dim_size = _infer_dim_size(index, dim)
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    if src.dtype.is_floating_point:
        out = torch.full(out_shape, torch.inf, dtype=src.dtype, device=src.device)
    else:
        out = torch.full(out_shape, torch.iinfo(src.dtype).max, dtype=src.dtype, device=src.device)
    out.scatter_reduce_(dim, index, src, reduce="amin", include_self=True)
    return out, None


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int | None = None,
) -> torch.Tensor:
    dim = _normalize_dim(dim, src.dim())
    index = _expand_index(index, src)
    max_vals, _ = scatter_max(src, index, dim=dim, dim_size=dim_size)
    gathered_max = max_vals.gather(dim, index)
    exp = torch.exp(src - gathered_max)
    denom = scatter_add(exp, index, dim=dim, dim_size=dim_size)
    gathered_denom = denom.gather(dim, index).clamp_min(1e-12)
    return exp / gathered_denom
