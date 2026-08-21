"""Realtime 2D and 3D detection image rendering."""

import os
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from pipeline import VIS_PANEL_ASPECT
from pipeline.output_writer import ScenePublishGate
from utils.image import draw_bb3s, put_text, render_bb2


def semantic_name_cn(label: str, vocabulary: dict) -> str:
    """Map one English semantic label to Chinese."""
    return vocabulary["en_to_cn"].get(str(label), str(label))


def semantic_color_bgr(label: str, vocabulary: dict) -> tuple[int, int, int]:
    """Return one configured semantic color in OpenCV BGR order."""
    red, green, blue = vocabulary["en_to_rgb"][str(label)]
    return int(blue), int(green), int(red)


def scale_bb2d_xxyy(
    bb2d: torch.Tensor,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> torch.Tensor:
    """Scale Boxer-format 2DBB into the original image dimensions."""
    if bb2d is None or len(bb2d) == 0:
        return bb2d
    scaled = bb2d.detach().cpu().clone()
    scaled[:, [0, 1]] *= float(dst_width) / max(float(src_width), 1.0)
    scaled[:, [2, 3]] *= float(dst_height) / max(float(src_height), 1.0)
    return scaled


def fit_panel_16x9(panel: np.ndarray, width: int) -> np.ndarray:
    """Fit one image into a 16:9 canvas without changing its aspect ratio."""
    output_width = int(width)
    output_height = max(1, int(round(output_width / VIS_PANEL_ASPECT)))
    panel_height, panel_width = panel.shape[:2]
    scale = min(
        output_width / max(panel_width, 1),
        output_height / max(panel_height, 1),
    )
    resized_width = max(1, int(round(panel_width * scale)))
    resized_height = max(1, int(round(panel_height * scale)))
    resized = cv2.resize(
        panel,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((output_height, output_width, 3), 255, dtype=np.uint8)
    x_offset = (output_width - resized_width) // 2
    y_offset = (output_height - resized_height) // 2
    canvas[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ] = resized
    return canvas


def render_detection_panels(
    *,
    img_np: np.ndarray,
    orig_img_np: np.ndarray,
    bb2d: torch.Tensor,
    labels2d: Sequence[str],
    obb_pr_w: Any,
    labels3d: Sequence[str],
    T_wr: Any,
    cam: Any,
    boxernet: Any,
    frame_id: int,
    time_ns: int,
    vocabulary: dict,
    bb3_source_2d_indices: torch.Tensor,
    detection_line_thickness: int,
    detection_label_font_size: float,
    semantic_colors: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Render raw DINO 2DBB and final validated Boxer 3DBB panels."""
    resized_height, resized_width = img_np.shape[:2]
    original_height, original_width = orig_img_np.shape[:2]
    bb2d_vis = scale_bb2d_xxyy(
        bb2d,
        resized_width,
        resized_height,
        original_width,
        original_height,
    )
    camera_vis = cam.scale(
        (
            float(original_width) / max(float(resized_width), 1.0),
            float(original_height) / max(float(resized_height), 1.0),
        )
    )
    text_2d = [semantic_name_cn(label, vocabulary) for label in labels2d]
    text_3d = [semantic_name_cn(label, vocabulary) for label in labels3d]
    colors_2d = [semantic_color_bgr(label, vocabulary) for label in labels2d]
    colors_3d = [semantic_color_bgr(label, vocabulary) for label in labels3d]
    if not semantic_colors:
        colors_2d = [(0, 255, 255)] * len(labels2d)
        colors_3d = [
            colors_2d[int(source_index)]
            for source_index in bb3_source_2d_indices
        ]

    image_2d = render_bb2(
        orig_img_np.copy(),
        bb2d_vis,
        scale=detection_line_thickness,
        rotated=False,
        texts=text_2d,
        clr=colors_2d,
        text_sz=detection_label_font_size,
    )
    put_text(image_2d, "2D Detections (Grounding DINO)", scale=0.6, line=0)
    put_text(
        image_2d,
        f"frame {frame_id}, t={time_ns / 1e9:.3f}s",
        scale=0.5,
        line=2,
    )
    image_3d = draw_bb3s(
        viz=orig_img_np.copy(),
        T_world_rig=T_wr,
        cam=camera_vis,
        obbs=obb_pr_w,
        already_rotated=False,
        rotate_label=False,
        colors=colors_3d,
        texts=text_3d,
        text_sz=detection_label_font_size,
        thickness=detection_line_thickness,
    )
    put_text(
        image_3d,
        f"3D Detections (Boxer {boxernet.hw}x{boxernet.hw})",
        scale=0.6,
        line=0,
    )
    return (
        fit_panel_16x9(image_2d, original_width),
        fit_panel_16x9(image_3d, original_width),
    )


def write_detection_images(
    *,
    output_2d_path: Path,
    output_3d_path: Path,
    publish_gate: ScenePublishGate,
    **render_kwargs: Any,
) -> bool:
    """Publish a timestamp-matched image pair through temporary files."""
    image_2d, image_3d = render_detection_panels(**render_kwargs)
    outputs = (
        (image_2d, str(output_2d_path)),
        (image_3d, str(output_3d_path)),
    )
    encoded = []
    for image, output_path in outputs:
        success, jpeg = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not success:
            raise RuntimeError(f"failed to encode visualization: {output_path}")
        encoded.append((jpeg.tobytes(), output_path, f"{output_path}.tmp"))

    try:
        for image_bytes, _, temp_path in encoded:
            with open(temp_path, "wb") as output_file:
                output_file.write(image_bytes)

        def commit_pair() -> None:
            """Atomically replace both final images under the scene gate."""
            published = []
            try:
                for _, output_path, temp_path in encoded:
                    os.replace(temp_path, output_path)
                    published.append(output_path)
            except Exception:
                for output_path in published:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                raise

        return publish_gate.publish(commit_pair)
    finally:
        for _, _, temp_path in encoded:
            if os.path.exists(temp_path):
                os.remove(temp_path)
