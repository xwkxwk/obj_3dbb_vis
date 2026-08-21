"""GroundingDINO detection helpers for the unified pipeline"""

import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any, Optional

import numpy as np
import torch
from loguru import logger

from pipeline import PROJECT_ROOT
from filter.depth_filter import filter_2dbb_by_patch_depth

DetectionResult = tuple[
    torch.Tensor,
    torch.Tensor,
    list[str],
    torch.Tensor,
    torch.Tensor,
    list[str],
    torch.Tensor,
]


def nms_2d_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Return score-ordered indices retained by 2D NMS."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        inter_x1 = np.maximum(x1[current], x1[rest])
        inter_y1 = np.maximum(y1[current], y1[rest])
        inter_x2 = np.minimum(x2[current], x2[rest])
        inter_y2 = np.minimum(y2[current], y2[rest])
        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        union = areas[current] + areas[rest] - inter
        iou = np.divide(
            inter,
            union,
            out=np.zeros_like(inter, dtype=np.float32),
            where=union > 0,
        )
        order = rest[iou <= iou_threshold]

    return np.array(keep, dtype=np.int64)

def filter_boxes_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    source_indices: np.ndarray,
    width: int,
    height: int,
    min_area_ratio: float,
    margin: int,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Filter small and edge-touching raw GroundingDINO boxes."""
    if len(boxes) == 0:
        return boxes, scores, labels, source_indices

    keep = np.ones(len(boxes), dtype=bool)
    min_area = float(min_area_ratio) * width * height
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        if (x2 - x1) * (y2 - y1) < min_area:
            keep[idx] = False
        elif x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin:
            keep[idx] = False

    boxes = boxes[keep].copy()
    scores = scores[keep]
    source_indices = source_indices[keep]
    labels = [labels[idx] for idx in range(len(labels)) if keep[idx]]
    if len(boxes) > 0:
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    return boxes, scores, labels, source_indices

def conda_env_python(env_name: Optional[str]) -> Optional[str]:
    """Locate the Python executable of one unpacked Conda environment."""
    if not env_name:
        return None

    exe_name = "python.exe" if os.name == "nt" else "python"
    sub_path = Path(exe_name) if os.name == "nt" else Path("bin") / exe_name
    candidates = []

    active_prefix = os.environ.get("CONDA_PREFIX")
    if active_prefix:
        active = Path(active_prefix)
        if active.parent.name == "envs":
            candidates.append(active.parent / env_name / sub_path)
        candidates.append(active.parent / env_name / sub_path)

    conda_envs_path = os.environ.get("CONDA_ENVS_PATH")
    if conda_envs_path:
        for env_root in conda_envs_path.split(os.pathsep):
            if env_root:
                candidates.append(Path(env_root) / env_name / sub_path)

    home = Path.home()
    for root in (
        home / "anaconda3" / "envs",
        home / "miniconda3" / "envs",
        home / "Anaconda3" / "envs",
        home / "Miniconda3" / "envs",
    ):
        candidates.append(root / env_name / sub_path)

    if os.name == "nt":
        for root in (Path("E:/Anaconda_envs/envs"), Path("D:/Anaconda_envs/envs")):
            candidates.append(root / env_name / sub_path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None

def build_groundingdino_command(gdino_cfg: dict) -> list[str]:
    """Build the persistent GroundingDINO subprocess command."""
    server_script = PROJECT_ROOT / "src" / "groundingdino_server.py"
    python_path = gdino_cfg.get("python")
    env_name = gdino_cfg.get("conda_env")

    if python_path:
        cmd = [os.path.expanduser(python_path), str(server_script)]
    else:
        gdino_python = conda_env_python(env_name)
        if gdino_python:
            cmd = [gdino_python, str(server_script)]
        else:
            cmd = [
                "conda",
                "run",
                "-n",
                env_name,
                "--no-capture-output",
                "python",
                str(server_script),
            ]

    cmd.extend(
        [
            "--device",
            str(gdino_cfg.get("device", "cuda:0")),
            "--box-threshold",
            str(gdino_cfg["box_threshold"]),
            "--text-threshold",
            str(gdino_cfg["text_threshold"]),
            "--topk",
            str(gdino_cfg["topk"]),
            "--prompt",
            str(gdino_cfg["prompt"]),
        ]
    )
    if gdino_cfg.get("checkpoint"):
        cmd.extend(["--checkpoint", str(gdino_cfg["checkpoint"])])
    if gdino_cfg.get("bert_path"):
        cmd.extend(["--bert-path", str(gdino_cfg["bert_path"])])
    if gdino_cfg.get("config"):
        cmd.extend(["--config", str(gdino_cfg["config"])])
    if gdino_cfg.get("source_dir"):
        cmd.extend(["--groundingdino-dir", str(gdino_cfg["source_dir"])])
    return cmd

def start_groundingdino_server(gdino_cfg: dict) -> subprocess.Popen:
    """Start GroundingDINO and wait until its model is ready."""
    cmd = build_groundingdino_command(gdino_cfg)
    logger.debug("Spawning Grounding DINO server: {}", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    ready_line = proc.stdout.readline()
    if ready_line is None or ready_line.strip() != "READY":
        rc = proc.poll()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
            rc = proc.returncode
        stderr_tail = proc.stderr.read()[-4000:] if proc.stderr else ""
        raise RuntimeError(
            f"Grounding DINO server failed to start (exit code {rc}).\n"
            f"stdout: {ready_line}\nstderr tail:\n{stderr_tail}"
        )

    def drain_stderr() -> None:
        """Drain child diagnostics without mixing them into the JSON protocol."""
        for line in proc.stderr:
            message = line.rstrip()
            if message:
                logger.debug("GroundingDINO: {}", message)

    threading.Thread(target=drain_stderr, daemon=True).start()
    logger.debug("Grounding DINO server ready")
    return proc

def query_groundingdino(proc: subprocess.Popen, image_path: str) -> dict:
    """Request one detection result from the persistent child process."""
    proc.stdin.write(str(image_path) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        rc = proc.poll()
        if rc is not None:
            remaining = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(
                f"Grounding DINO server died (exit code {rc}).\n"
                f"stderr tail:\n{remaining[-2000:]}"
            )
        raise RuntimeError("Grounding DINO server closed stdout unexpectedly")
    result = json.loads(line)
    if "error" in result:
        raise RuntimeError(f"Grounding DINO detection error: {result['error']}")
    return result

def stop_groundingdino_server(
    proc: Optional[subprocess.Popen], timeout: float = 10.0
) -> bool:
    """Stop the persistent GroundingDINO subprocess."""
    if proc is None or proc.poll() is not None:
        return True
    try:
        proc.stdin.write("exit\n")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(timeout=max(float(timeout), 0.0))
        return True
    except Exception:
        proc.kill()
        proc.wait()
        return False

def detect_groundingdino(
    proc: subprocess.Popen,
    datum: dict,
    img_np: np.ndarray,
    image_path: str,
    gdino_cfg: dict,
    boxernet: Any,
    depth_filter_min_m: float,
    depth_filter_max_m: float,
) -> DetectionResult:
    """Run GroundingDINO and return raw and depth-filtered 2D boxes."""
    height, width = img_np.shape[:2]
    result = query_groundingdino(proc, image_path)
    raw_bb2d_np = np.array(result["boxes"], dtype=np.float32)
    raw_scores2d_np = np.array(result["scores"], dtype=np.float32)
    raw_labels2d = list(result["labels"])
    orig_w = int(result.get("img_width", width))
    orig_h = int(result.get("img_height", height))

    if len(raw_bb2d_np) > 0:
        raw_bb2d_vis_np = raw_bb2d_np.copy()
        if orig_w > 0 and orig_h > 0 and (orig_w != width or orig_h != height):
            raw_bb2d_vis_np[:, [0, 2]] *= width / orig_w
            raw_bb2d_vis_np[:, [1, 3]] *= height / orig_h
        raw_bb2d_vis_np[:, [0, 2]] = np.clip(
            raw_bb2d_vis_np[:, [0, 2]], 0, width - 1
        )
        raw_bb2d_vis_np[:, [1, 3]] = np.clip(
            raw_bb2d_vis_np[:, [1, 3]], 0, height - 1
        )
        raw_bb2d = torch.from_numpy(raw_bb2d_vis_np[:, [0, 2, 1, 3]]).float()
        raw_scores2d = torch.from_numpy(raw_scores2d_np).float()
    else:
        raw_bb2d = torch.zeros(0, 4)
        raw_scores2d = torch.zeros(0)

    bb2d_np = raw_bb2d_np.copy()
    scores2d_np = raw_scores2d_np.copy()
    labels2d = list(raw_labels2d)
    source_indices = np.arange(len(raw_bb2d_np), dtype=np.int64)

    if len(bb2d_np) > 0:
        bb2d_np, scores2d_np, labels2d, source_indices = filter_boxes_xyxy(
            bb2d_np,
            scores2d_np,
            labels2d,
            source_indices,
            orig_w,
            orig_h,
            gdino_cfg["min_area_ratio"],
            gdino_cfg["margin"],
        )
        if gdino_cfg["nms_iou"] > 0 and len(bb2d_np) > 0:
            keep = nms_2d_xyxy(bb2d_np, scores2d_np, iou_threshold=gdino_cfg["nms_iou"])
            bb2d_np = bb2d_np[keep]
            scores2d_np = scores2d_np[keep]
            source_indices = source_indices[keep]
            labels2d = [labels2d[idx] for idx in keep]

        if len(bb2d_np) > 0 and orig_w > 0 and orig_h > 0 and (orig_w != width or orig_h != height):
            bb2d_np[:, [0, 2]] *= width / orig_w
            bb2d_np[:, [1, 3]] *= height / orig_h
            bb2d_np[:, [0, 2]] = np.clip(bb2d_np[:, [0, 2]], 0, width - 1)
            bb2d_np[:, [1, 3]] = np.clip(bb2d_np[:, [1, 3]], 0, height - 1)

    if len(bb2d_np) == 0:
        return (
            torch.zeros(0, 4),
            torch.zeros(0),
            [],
            raw_bb2d,
            raw_scores2d,
            raw_labels2d,
            torch.zeros(0, dtype=torch.long),
        )

    bb2d = torch.from_numpy(bb2d_np[:, [0, 2, 1, 3]]).float()
    scores2d = torch.from_numpy(scores2d_np).float()
    bb2d, scores2d, labels2d, depth_keep = filter_2dbb_by_patch_depth(
        bb2d,
        scores2d,
        labels2d,
        datum["sdp_w"].float(),
        datum["cam0"].float(),
        datum["T_world_rig0"].float(),
        height,
        width,
        boxernet.dino.patch_size,
        depth_filter_min_m,
        depth_filter_max_m,
        return_keep=True,
    )
    source_2d_indices = torch.from_numpy(source_indices).long()[depth_keep.cpu()]
    return (
        bb2d,
        scores2d,
        labels2d,
        raw_bb2d,
        raw_scores2d,
        raw_labels2d,
        source_2d_indices,
    )
