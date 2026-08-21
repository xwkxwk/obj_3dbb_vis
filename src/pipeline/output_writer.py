"""Realtime Fuse CSV output helpers."""

import csv
import os
import threading
from typing import Callable, Optional

import numpy as np
from loguru import logger

from pipeline import R_ALIGN_INV
from pipeline.config import vocabulary_name_cn


class ScenePublishGate:
    """Serialize final file publication and cancellation for one scene."""

    def __init__(self) -> None:
        """Create one active per-scene publication gate."""
        self.cancelled = threading.Event()
        self.lock = threading.Lock()

    def cancel(self) -> None:
        """Prevent all future final-path publications for this scene."""
        with self.lock:
            self.cancelled.set()

    def is_cancelled(self) -> bool:
        """Return whether this scene no longer accepts publications."""
        return self.cancelled.is_set()

    def publish(self, callback: Callable[[], None]) -> bool:
        """Run one final-path commit unless the scene was cancelled."""
        with self.lock:
            if self.cancelled.is_set():
                return False
            callback()
            return True


def pipeline_log(
    log_callback: Optional[Callable[[str], None]], message: str
) -> None:
    """Write one pipeline message through the optional service callback."""
    if log_callback is not None:
        log_callback(message)
    else:
        logger.debug(message)


def quaternion_wxyz_to_matrix(
    qw: float, qx: float, qy: float, qz: float
) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    quaternion = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm <= 0:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = quaternion / norm
    return np.asarray(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    rotation = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(
            1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
        ) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0, 1] + rotation[1, 0]) / scale
        qz = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(
            1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
        ) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / scale
        qx = (rotation[0, 1] + rotation[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(
            1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
        ) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / scale
        qx = (rotation[0, 2] + rotation[2, 0]) / scale
        qy = (rotation[1, 2] + rotation[2, 1]) / scale
        qz = 0.25 * scale
    quaternion = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    return quaternion


def write_sfm_fused_csv(
    fused_csv_path: str,
    vocabulary: dict,
    semantic_colors: bool = False,
) -> Optional[str]:
    """Write one fused CSV in SFM coordinates with Chinese names and colors."""
    if not fused_csv_path or not os.path.exists(fused_csv_path):
        return None
    sfm_csv_path = os.path.splitext(fused_csv_path)[0] + "_sfm.csv"
    with open(fused_csv_path, "r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        fieldnames.extend(
            name for name in ("r", "g", "b") if name not in fieldnames
        )
        rows = []
        for row in reader:
            translation = np.asarray(
                [
                    float(row["tx_world_object"]),
                    float(row["ty_world_object"]),
                    float(row["tz_world_object"]),
                ],
                dtype=np.float64,
            )
            rotation = quaternion_wxyz_to_matrix(
                float(row["qw_world_object"]),
                float(row["qx_world_object"]),
                float(row["qy_world_object"]),
                float(row["qz_world_object"]),
            )
            translation_sfm = R_ALIGN_INV @ translation
            quaternion_sfm = matrix_to_quaternion_wxyz(R_ALIGN_INV @ rotation)
            row["tx_world_object"] = str(translation_sfm[0])
            row["ty_world_object"] = str(translation_sfm[1])
            row["tz_world_object"] = str(translation_sfm[2])
            row["qw_world_object"] = str(quaternion_sfm[0])
            row["qx_world_object"] = str(quaternion_sfm[1])
            row["qy_world_object"] = str(quaternion_sfm[2])
            row["qz_world_object"] = str(quaternion_sfm[3])
            semantic_name = row["name"]
            row["name"] = vocabulary_name_cn(semantic_name, vocabulary)
            red, green, blue = (
                vocabulary["en_to_rgb"][semantic_name]
                if semantic_colors
                else _stable_instance_color(int(row["fused_instance"]))
            )
            row["r"] = str(red)
            row["g"] = str(green)
            row["b"] = str(blue)
            rows.append(row)

    with open(sfm_csv_path, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return sfm_csv_path


def _stable_instance_color(fused_instance: int) -> tuple[int, int, int]:
    """Return a stable fallback RGB color for one fused instance."""
    colors = [
        (254, 253, 90),
        (233, 62, 227),
        (251, 249, 116),
        (80, 176, 255),
        (100, 220, 120),
        (255, 140, 80),
        (180, 120, 255),
        (255, 90, 120),
        (90, 220, 220),
        (220, 180, 80),
    ]
    if fused_instance < len(colors):
        return colors[fused_instance]
    return (
        int((fused_instance * 73 + 97) % 256),
        int((fused_instance * 151 + 53) % 256),
        int((fused_instance * 199 + 211) % 256),
    )
