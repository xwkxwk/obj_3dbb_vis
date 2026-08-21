import torch
from typing import Any, Sequence

from boxernet.boxernet import sdp_to_patches


def _batched_wrapper(wrapper: Any) -> Any:
    """Add a batch dimension to one Boxer tensor wrapper when needed."""
    data = wrapper._data
    if data.ndim == 1:
        return wrapper.__class__(data.unsqueeze(0))
    return wrapper


@torch.no_grad()
def filter_2dbb_by_patch_depth(
    bb2d: torch.Tensor,
    scores2d: torch.Tensor,
    labels2d: Sequence[str],
    sdp_w: torch.Tensor,
    cam: Any,
    T_wr: Any,
    H: int,
    W: int,
    patch_size: Any,
    min_depth_m: float = 0.4,
    max_depth_m: float = 8.0,
    return_keep: bool = False,
) -> tuple:
    """过滤深度完全位于上下限之外的2DBB"""
    if (min_depth_m <= 0 and max_depth_m <= 0) or bb2d.shape[0] == 0:
        if return_keep:
            return bb2d, scores2d, labels2d, torch.ones(bb2d.shape[0], dtype=torch.bool, device=bb2d.device)
        return bb2d, scores2d, labels2d
    if sdp_w is None or sdp_w.numel() == 0:
        if return_keep:
            return bb2d, scores2d, labels2d, torch.ones(bb2d.shape[0], dtype=torch.bool, device=bb2d.device)
        return bb2d, scores2d, labels2d

    patch_size = int(patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size)
    sdp_w = sdp_w.float()
    if sdp_w.ndim == 2:
        sdp_w = sdp_w.unsqueeze(0)
    cam = _batched_wrapper(cam.float())
    T_wr = _batched_wrapper(T_wr.float())

    sdp_patch = sdp_to_patches(sdp_w, cam, T_wr, int(H), int(W), patch_size)[0, 0]
    fH, fW = sdp_patch.shape
    patch_h = float(H) / float(fH)
    patch_w = float(W) / float(fW)

    keep = []
    for box in bb2d.detach().cpu():
        x1 = min(float(box[0]), float(box[1]))
        x2 = max(float(box[0]), float(box[1]))
        y1 = min(float(box[2]), float(box[3]))
        y2 = max(float(box[2]), float(box[3]))

        px1 = max(0, min(fW - 1, int(x1 // patch_w)))
        px2 = max(0, min(fW - 1, int(x2 // patch_w)))
        py1 = max(0, min(fH - 1, int(y1 // patch_h)))
        py2 = max(0, min(fH - 1, int(y2 // patch_h)))

        depths = sdp_patch[py1 : py2 + 1, px1 : px2 + 1].reshape(-1)
        valid = torch.isfinite(depths) & (depths > 0)
        too_near = bool(min_depth_m > 0 and torch.all(depths < min_depth_m))
        too_far = bool(max_depth_m > 0 and torch.all(depths > max_depth_m))
        remove = bool(valid.all()) and (too_near or too_far)
        keep.append(not remove)

    keep_mask = torch.tensor(keep, dtype=torch.bool, device=bb2d.device)
    result = (
        bb2d[keep_mask],
        scores2d[keep_mask],
        [label for label, keep_one in zip(labels2d, keep) if keep_one],
    )
    if return_keep:
        return result + (keep_mask,)
    return result
