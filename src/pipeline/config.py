"""Realtime visualization configuration helpers."""

import importlib.util
import re
from pathlib import Path
from typing import Any

import torch

from pipeline import PROJECT_ROOT


def _get(config: dict, path: str, default: Any = None) -> Any:
    """Read one nested configuration value."""
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def resolve_path(value: Any) -> str:
    """Resolve one project-relative path."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def load_vocabulary(config_path: str, config: dict) -> dict:
    """Load and validate the configured bilingual vocabulary."""
    vocabulary_path = Path(config_path).resolve().parent / "vocabulary.py"
    spec = importlib.util.spec_from_file_location(
        "obj3dbb_vis_vocabulary", str(vocabulary_path)
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load vocabulary: {vocabulary_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    vocabularies = module.VOCABULARIES
    vocabulary_name = str(config.get("vocabulary", module.DEFAULT_VOCABULARY))
    if vocabulary_name not in vocabularies:
        raise ValueError(f"vocabulary not found: {vocabulary_name}")
    vocabulary = vocabularies[vocabulary_name]
    labels_en = [str(value) for value in vocabulary.get("en", [])]
    labels_cn = [str(value) for value in vocabulary.get("cn", [])]
    colors = vocabulary.get("colors", {})
    if len(labels_en) != len(labels_cn) or set(colors) != set(labels_en):
        raise ValueError("vocabulary labels and colors must match")

    en_to_rgb = {}
    for label in labels_en:
        color = colors[label]
        if not isinstance(color, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", color) is None:
            raise ValueError(f"invalid vocabulary color: {label}={color}")
        en_to_rgb[label] = tuple(
            int(color[index:index + 2], 16) for index in (1, 3, 5)
        )
    prompt = vocabulary.get("prompt") or " . ".join(labels_en) + " ."
    return {
        "labels_en": labels_en,
        "prompt": str(prompt),
        "en_to_cn": dict(zip(labels_en, labels_cn)),
        "en_to_rgb": en_to_rgb,
    }


def vocabulary_name_cn(name: str, vocabulary: dict) -> str:
    """Map one English semantic label to Chinese."""
    return vocabulary.get("en_to_cn", {}).get(str(name), str(name))


def build_realtime_config(config: dict, config_path: str) -> dict:
    """Build the camera-independent realtime processor configuration."""
    boxer_config = config.get("boxer", {})
    detector_config = _get(config, "pipelines.groundingdino", {})
    fuse_config = config.get("fuse", {})
    visualization_config = config.get("visualization", {})
    validation_config = config.get("validation", {})
    boxer_model = _get(config, "models.boxer", {})
    groundingdino_model = _get(config, "models.groundingdino", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocabulary = load_vocabulary(config_path, config)

    return {
        "device": device,
        "force_precision": None,
        "checkpoint": resolve_path(boxer_model["checkpoint"]),
        "num_samples": int(boxer_config.get("num_samples", 100000)),
        "thresh3d": float(boxer_config.get("thresh3d", 0.5)),
        "depth_filter_min_m": float(boxer_config.get("depth_filter_min_m", 0.4)),
        "depth_filter_max_m": float(boxer_config.get("depth_filter_max_m", 8.0)),
        "visualization": {
            "detection_line_thickness": int(
                visualization_config.get("detection_line_thickness", 2)
            ),
            "detection_label_font_size": float(
                visualization_config.get("detection_label_font_size", 0.8)
            ),
        },
        "validation": {
            "projected_2d_iou": {
                "enabled": bool(
                    _get(validation_config, "projected_2d_iou.enabled", True)
                ),
                "min_iou": float(
                    _get(validation_config, "projected_2d_iou.min_iou", 0.5)
                ),
            }
        },
        "fuse": {
            "cluster_iou_threshold": float(
                fuse_config.get("cluster_iou_threshold", 0.2)
            ),
            "min_detections": int(fuse_config.get("min_detections", 4)),
            "conf_threshold": float(fuse_config.get("conf_threshold", 0.55)),
            "semantic_threshold": float(
                fuse_config.get("semantic_threshold", 0.0)
            ),
            "nms": bool(fuse_config.get("nms", True)),
            "nms_iou": float(fuse_config.get("nms_iou", 0.4)),
            "same_semantic_ios_absorb": bool(
                fuse_config.get("same_semantic_ios_absorb", True)
            ),
            "ios_absorb_threshold": float(
                fuse_config.get("ios_absorb_threshold", 0.8)
            ),
        },
        "groundingdino": {
            "source_dir": resolve_path(
                groundingdino_model.get("source_dir", "libs/GroundingDINO")
            ),
            "config": resolve_path(groundingdino_model["config"]),
            "checkpoint": resolve_path(groundingdino_model["checkpoint"]),
            "bert_path": resolve_path(groundingdino_model["bert_path"]),
            "conda_env": "xwk_gdino",
            "python": None,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "prompt": vocabulary["prompt"],
            "box_threshold": float(detector_config.get("box_threshold", 0.3)),
            "text_threshold": float(detector_config.get("text_threshold", 0.25)),
            "topk": int(detector_config.get("topk", 100)),
            "min_area_ratio": float(detector_config.get("min_area_ratio", 0.002)),
            "margin": int(detector_config.get("margin", 3)),
            "nms_iou": float(detector_config.get("nms_iou", 0.5)),
        },
        "vocabulary": vocabulary,
    }
