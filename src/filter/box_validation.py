import torch
from typing import Any, Sequence

from utils.tw.obb import bb2d_from_project_bb3d


def _sequence_wrapper(wrapper: Any) -> Any:
    """Add a sequence dimension to one Boxer tensor wrapper when needed."""
    data = wrapper._data
    if data.ndim == 1:
        return wrapper.__class__(data.unsqueeze(0))
    return wrapper


def _box_iou_xxyy(
    boxes_a: torch.Tensor, boxes_b: torch.Tensor
) -> torch.Tensor:
    """Calculate aligned IoU values for Boxer xxyy boxes."""
    x1 = torch.maximum(boxes_a[:, 0], boxes_b[:, 0])
    y1 = torch.maximum(boxes_a[:, 2], boxes_b[:, 2])
    x2 = torch.minimum(boxes_a[:, 1], boxes_b[:, 1])
    y2 = torch.minimum(boxes_a[:, 3], boxes_b[:, 3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_a = torch.clamp(boxes_a[:, 1] - boxes_a[:, 0], min=0) * torch.clamp(
        boxes_a[:, 3] - boxes_a[:, 2], min=0
    )
    area_b = torch.clamp(boxes_b[:, 1] - boxes_b[:, 0], min=0) * torch.clamp(
        boxes_b[:, 3] - boxes_b[:, 2], min=0
    )
    union = area_a + area_b - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


@torch.no_grad()
def filter_3dbb_by_validation(
    obb_pr_w: Any,
    bb2d_for_3d: torch.Tensor,
    scores3d: torch.Tensor,
    labels3d: Sequence[str],
    sdp_w: torch.Tensor,
    cam: Any,
    T_wr: Any,
    H: int,
    W: int,
    patch_size: Any,
    validation_cfg: dict,
    return_stats: bool = False,
    return_keep: bool = False,
) -> tuple:
    """Filter 3DBB by projected 2D IoU and optionally return stats and mask."""
    stats = {
        "input_3dbb": int(obb_pr_w.shape[0]),
        "projected_2d_iou_removed": 0,
        "kept_3dbb": int(obb_pr_w.shape[0]),
    }
    if obb_pr_w.shape[0] == 0:
        keep = torch.ones(obb_pr_w.shape[0], dtype=torch.bool)
        if return_stats:
            if return_keep:
                return obb_pr_w, scores3d, labels3d, stats, keep
            return obb_pr_w, scores3d, labels3d, stats
        if return_keep:
            return obb_pr_w, scores3d, labels3d, keep
        return obb_pr_w, scores3d, labels3d

    keep = torch.ones(obb_pr_w.shape[0], dtype=torch.bool)

    iou_cfg = validation_cfg.get("projected_2d_iou", {})
    if iou_cfg.get("enabled", True):
        cam_seq = _sequence_wrapper(cam.float())
        T_wr_seq = _sequence_wrapper(T_wr.float())
        projected_bb2d, projected_valid = bb2d_from_project_bb3d(
            obb_pr_w.float(),
            cam_seq,
            T_wr_seq,
        )
        projected_bb2d = projected_bb2d.squeeze(0).cpu()
        projected_valid = projected_valid.squeeze(0).cpu()
        target_bb2d = bb2d_for_3d.detach().cpu().float()
        ious = _box_iou_xxyy(projected_bb2d, target_bb2d)
        iou_keep = projected_valid & (ious > float(iou_cfg.get("min_iou", 0.7)))
        stats["projected_2d_iou_removed"] = int((keep & ~iou_keep).sum().item())
        keep &= iou_keep

    keep = keep.to(device=obb_pr_w._data.device)
    stats["kept_3dbb"] = int(keep.sum().item())
    result = (
        obb_pr_w[keep].clone(),
        scores3d[keep].clone(),
        [label for label, keep_one in zip(labels3d, keep.cpu().tolist()) if keep_one],
    )
    if return_stats:
        if return_keep:
            return (*result, stats, keep.cpu())
        return (*result, stats)
    if return_keep:
        return (*result, keep.cpu())
    return result
