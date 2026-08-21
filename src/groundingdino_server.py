#! /usr/bin/env python3
"""Grounding DINO inference server for Boxer real-time pipelines.

The server loads Grounding DINO once, then reads image paths from stdin and
writes one JSON result per image to stdout. Keep stdout clean: the Boxer parent
process treats each stdout line as protocol data.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from PIL import Image


DEFAULT_PROMPT = (
    "chair . desk . box . basket . monitor . screen . display . tv . "
    "laptop . keyboard . mouse . bottle . robot . "
)


def repo_root() -> Path:
    """Return the mounted Boxer project root."""
    return Path(__file__).resolve().parents[1]


def find_groundingdino_dir(root: Path, override: Optional[str]) -> Path:
    """Resolve the GroundingDINO source directory from an override or image paths."""
    if override:
        return Path(override).expanduser().resolve()

    candidates = [
        root / "libs" / "GroundingDINO",
        root / "GroundingDINO",
        root / "groundingdino",
        root / "Grounding-DINO",
    ]
    for candidate in candidates:
        if (candidate / "groundingdino").exists():
            return candidate
    return candidates[0]


def resolve_path(path_value: str, root: Path, gdino_dir: Path) -> Path:
    """Resolve one absolute, project-relative or GroundingDINO-relative path."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path

    root_path = root / path
    if root_path.exists():
        return root_path

    gdino_path = gdino_dir / path
    if gdino_path.exists():
        return gdino_path

    return root_path


def read_prompt(value: str) -> str:
    """Read a prompt value or prompt text file and normalize it."""
    path = Path(value).expanduser()
    if path.exists():
        parts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        return normalize_prompt(parts)

    if "," in value and "." not in value:
        return normalize_prompt([part.strip() for part in value.split(",")])
    return normalize_prompt(value)


def normalize_prompt(value: Any) -> str:
    """Normalize prompt phrases into GroundingDINO caption syntax."""
    if isinstance(value, (list, tuple)):
        caption = " . ".join(str(item).strip() for item in value if str(item).strip())
    else:
        caption = str(value).strip()

    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    return caption


def import_groundingdino(gdino_dir: Path) -> tuple[Any, Any, Any, Any]:
    """Import GroundingDINO modules from the base image source tree."""
    sys.path.insert(0, str(gdino_dir))
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

    return T, build_model, SLConfig, clean_state_dict


def load_image(
    image_path: Path, transforms_module: Any
) -> Tuple[Image.Image, torch.Tensor]:
    """Load and transform one RGB image for GroundingDINO."""
    image_pil = Image.open(image_path).convert("RGB")
    transform = transforms_module.Compose(
        [
            transforms_module.RandomResize([800], max_size=1333),
            transforms_module.ToTensor(),
            transforms_module.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image_pil, None)
    return image_pil, image_tensor


def load_model(
    config_path: Path,
    checkpoint_path: Path,
    bert_path: Path,
    device: str,
    build_model: Any,
    SLConfig: Any,
    clean_state_dict: Any,
) -> Any:
    """Construct GroundingDINO and load its checkpoint once."""
    args = SLConfig.fromfile(str(config_path))
    args.device = device
    args.text_encoder_type = str(bert_path)
    model = build_model(args)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model.to(device)


def prompt_phrase_spans(caption: str) -> List[Tuple[str, int, int]]:
    """Split prompt into phrase text and character spans"""
    spans = []
    start = 0
    for part in caption.split("."):
        end = start + len(part)
        phrase = part.strip()
        if phrase:
            left = start + len(part) - len(part.lstrip())
            right = start + len(part.rstrip())
            spans.append((phrase, left, right))
        start = end + 1
    return spans


def top1_prompt_text(
    logit: torch.Tensor, caption: str, tokenizer: Any
) -> Tuple[str, float]:
    """Select the single prompt phrase with the highest token score"""
    tokenized = tokenizer(caption, return_offsets_mapping=True)
    offsets = tokenized["offset_mapping"]
    best_label = ""
    best_score = -1.0

    for phrase, left, right in prompt_phrase_spans(caption):
        token_indices = []
        for idx, (tok_left, tok_right) in enumerate(offsets):
            if idx >= logit.numel():
                break
            if tok_right <= tok_left:
                continue
            if tok_left < right and tok_right > left:
                token_indices.append(idx)

        if not token_indices:
            continue

        score = float(logit[token_indices].max().item())
        if score > best_score:
            best_label = phrase
            best_score = score

    if best_label:
        return best_label, best_score

    top_idx = int(logit.argmax().item())
    token_id = tokenized["input_ids"][top_idx]
    return tokenizer.decode([token_id]).replace(".", "").strip(), float(logit[top_idx].item())


@torch.no_grad()
def predict(
    model: Any,
    image: torch.Tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> tuple[torch.Tensor, list[float], list[str]]:
    """Run one GroundingDINO inference request."""
    image = image.to(device)
    outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0].cpu()
    boxes = outputs["pred_boxes"][0].cpu()
    keep = logits.max(dim=1)[0] > box_threshold
    logits = logits[keep]
    boxes = boxes[keep]

    tokenizer = model.tokenizer
    labels = []
    scores = []
    for logit in logits:
        label, score = top1_prompt_text(logit, caption, tokenizer)
        labels.append(label)
        scores.append(score)

    return boxes, scores, labels


def box_cxcywh_to_xyxy(box: torch.Tensor, width: int, height: int) -> List[float]:
    """Convert one normalized cxcywh box to clipped pixel xyxy coordinates."""
    cx, cy, box_w, box_h = [float(value) for value in box.tolist()]
    x0 = (cx - box_w / 2.0) * width
    y0 = (cy - box_h / 2.0) * height
    x1 = (cx + box_w / 2.0) * width
    y1 = (cy + box_h / 2.0) * height
    return [
        max(0.0, min(float(width - 1), x0)),
        max(0.0, min(float(height - 1), y0)),
        max(0.0, min(float(width - 1), x1)),
        max(0.0, min(float(height - 1), y1)),
    ]


def parse_args() -> argparse.Namespace:
    """Parse the stdin/stdout server command-line options."""
    parser = argparse.ArgumentParser(
        description="Grounding DINO stdin/stdout inference server."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--groundingdino-dir", default=None)
    parser.add_argument(
        "--config",
        default="groundingdino/config/GroundingDINO_SwinB_cfg.py",
        help="Grounding DINO config path, absolute or relative to GroundingDINO/.",
    )
    parser.add_argument(
        "--checkpoint",
        default="ckpts/groundingdino_swinb_cogcoor.pth",
        help="Grounding DINO checkpoint path, absolute or relative to GroundingDINO/.",
    )
    parser.add_argument(
        "--bert-path",
        default="ckpts/bert-base-uncased",
        help="Local bert-base-uncased path, absolute or relative to project root.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the model once and serve image paths from standard input."""
    args = parse_args()
    root = repo_root()
    gdino_dir = find_groundingdino_dir(root, args.groundingdino_dir)
    config_path = resolve_path(args.config, root, gdino_dir)
    checkpoint_path = resolve_path(args.checkpoint, root, gdino_dir)
    bert_path = resolve_path(args.bert_path, root, gdino_dir)
    if not bert_path.exists():
        raise FileNotFoundError(f"BERT path not found: {bert_path}")
    if not (bert_path / "config.json").exists():
        raise FileNotFoundError(f"BERT config.json not found: {bert_path / 'config.json'}")
    prompt = read_prompt(args.prompt)

    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        (
            transforms_module,
            build_model,
            SLConfig,
            clean_state_dict,
        ) = import_groundingdino(gdino_dir)
        model = load_model(
            config_path,
            checkpoint_path,
            bert_path,
            args.device,
            build_model,
            SLConfig,
            clean_state_dict,
        )
    finally:
        sys.stdout = real_stdout

    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "exit":
            break

        image_path = Path(line).expanduser()
        if not image_path.exists():
            result = {"error": f"File not found: {line}"}
        else:
            try:
                image_pil, image_tensor = load_image(image_path, transforms_module)
                width, height = image_pil.size
                boxes_cxcywh, scores, labels = predict(
                    model,
                    image_tensor,
                    prompt,
                    args.box_threshold,
                    args.text_threshold,
                    args.device,
                )

                if len(scores) > args.topk:
                    order = sorted(
                        range(len(scores)), key=lambda idx: scores[idx], reverse=True
                    )[: args.topk]
                    boxes_cxcywh = boxes_cxcywh[order]
                    scores = [scores[idx] for idx in order]
                    labels = [labels[idx] for idx in order]

                boxes_xyxy = [
                    box_cxcywh_to_xyxy(box, width, height) for box in boxes_cxcywh
                ]
                result = {
                    "boxes": boxes_xyxy,
                    "scores": scores,
                    "labels": labels,
                    "img_width": width,
                    "img_height": height,
                }
            except Exception as exc:
                result = {"error": str(exc)}

        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
