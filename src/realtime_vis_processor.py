"""Realtime RGB-D visualization processor."""

import gc
import os
import pickle
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from loguru import logger

from pipeline import R_ALIGN
from boxernet.boxernet import BoxerNet
from filter.box_validation import filter_3dbb_by_validation
from pipeline.groundingdino_detector import (
    detect_groundingdino,
    start_groundingdino_server,
    stop_groundingdino_server,
)
from pipeline.incremental_fuse import FrameDetections, IncrementalFuseWorker
from pipeline.output_writer import ScenePublishGate
from pipeline.visualization import write_detection_images
from utils.image import torch2cv2
from utils.tw.camera import CameraTW
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW
from utils.tw.tensor_utils import pad_string, string2tensor


@dataclass
class RGBDFrame:
    """One decoded RGB-D frame received from Redis."""

    timestamp: int
    rgb_bytes: bytes
    rgb: np.ndarray
    rgb_bgr: np.ndarray
    depth_m: np.ndarray
    pose: tuple[float, float, float, float, float, float, float]


@dataclass
class RedisRGBDFrame:
    """One raw RGB-D frame extracted from a Redis message."""

    timestamp: int
    rgb_bytes: bytes
    depth_bytes: bytes
    camera_pose: bytes


@dataclass
class SceneRuntime:
    """Per-scene workers, paths, camera parameters and cancellation state."""

    output_2d_dir: Path
    output_3d_dir: Path
    intrinsics: np.ndarray
    width: int
    height: int
    temp_dir: str
    stop_event: threading.Event
    publish_gate: ScenePublishGate
    fuse_worker: Optional[IncrementalFuseWorker] = None
    frame_thread: Optional[threading.Thread] = None


class RealtimeVisProcessor:
    """Own models and process one realtime RGB-D scene."""

    def __init__(self, run_config: dict) -> None:
        """Load BoxerNet and start the persistent GroundingDINO process."""
        self.config = run_config
        self.device = run_config["device"]
        self.vocabulary = run_config["vocabulary"]
        self.boxernet = BoxerNet.load_from_checkpoint(
            run_config["checkpoint"], device=self.device
        )
        self.groundingdino = start_groundingdino_server(
            run_config["groundingdino"]
        )
        self.sem_name_to_id = {
            label: index
            for index, label in enumerate(self.vocabulary["labels_en"])
        }
        self.sem_id_to_name = {
            index: label for label, index in self.sem_name_to_id.items()
        }
        self.scene: Optional[SceneRuntime] = None
        self.frame_condition = threading.Condition()
        self.pending_frame = None
        self.frame_active = False
        self.fatal_error = None
        self.error_callback = None

    @staticmethod
    def decode_frame(
        frame: RedisRGBDFrame, width: int, height: int
    ) -> RGBDFrame:
        """Decode and validate one Redis RGB-D message."""
        rgb_bytes = frame.rgb_bytes
        depth_bytes = frame.depth_bytes
        rgb_bgr = cv2.imdecode(
            np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        depth_raw = cv2.imdecode(
            np.frombuffer(depth_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if rgb_bgr is None or depth_raw is None:
            raise ValueError("failed to decode RGB or depth image")
        if rgb_bgr.shape[:2] != (height, width) or depth_raw.shape != (height, width):
            raise ValueError(
                f"frame size mismatch: rgb={rgb_bgr.shape[:2]} "
                f"depth={depth_raw.shape} expected={(height, width)}"
            )
        if depth_raw.dtype != np.uint16:
            raise ValueError(f"depth image must be uint16 PNG, got {depth_raw.dtype}")

        pose_data = pickle.loads(frame.camera_pose)
        try:
            position = pose_data["position"]
            orientation = pose_data["orientation"]
            pose = (
                float(position["x"]),
                float(position["y"]),
                float(position["z"]),
                float(orientation["x"]),
                float(orientation["y"]),
                float(orientation["z"]),
                float(orientation["w"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid camera_pose structure") from exc
        if not np.isfinite(np.asarray(pose)).all():
            raise ValueError("camera_pose contains non-finite values")
        if np.linalg.norm(np.asarray(pose[3:], dtype=np.float64)) <= 1e-8:
            raise ValueError("camera_pose quaternion is invalid")

        return RGBDFrame(
            timestamp=frame.timestamp,
            rgb_bytes=rgb_bytes,
            rgb=cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
            rgb_bgr=rgb_bgr,
            depth_m=depth_raw.astype(np.float32) / 1000.0,
            pose=pose,
        )

    def pre_start(
        self,
        output_path: Path,
        intrinsics: list[float],
        width: int,
        height: int,
        error_callback: Callable[[Exception], None],
    ) -> None:
        """Initialize one scene output and incremental Fuse state."""
        output_root = output_path.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        result_dirs = [
            (output_root / name).resolve()
            for name in ("2D_detections", "3D_detections", "fused")
        ]
        for result_dir in result_dirs:
            if result_dir.parent != output_root:
                raise ValueError(f"invalid output directory: {result_dir}")
            if result_dir.exists():
                shutil.rmtree(result_dir)
            result_dir.mkdir()

        output_2d_dir, output_3d_dir, fused_dir = result_dirs
        self.error_callback = error_callback
        scene = SceneRuntime(
            output_2d_dir=output_2d_dir,
            output_3d_dir=output_3d_dir,
            intrinsics=np.asarray(intrinsics, dtype=np.float32).reshape(3, 3),
            width=int(width),
            height=int(height),
            temp_dir=tempfile.mkdtemp(prefix="obj3dbb_vis_"),
            stop_event=threading.Event(),
            publish_gate=ScenePublishGate(),
        )
        with self.frame_condition:
            self.pending_frame = None
            self.frame_active = False
            self.fatal_error = None
            self.scene = scene
        scene.fuse_worker = IncrementalFuseWorker(
            self.config["fuse"],
            self.device,
            str(fused_dir),
            self.vocabulary,
            scene.publish_gate,
            logger.debug,
            error_callback=lambda error: self._record_fatal(error, scene),
        )
        scene.frame_thread = threading.Thread(
            target=self._frame_loop,
            args=(scene,),
            name="obj3dbb-frame",
            daemon=True,
        )
        scene.frame_thread.start()

    def submit_frame(self, frame_id: int, frame: RedisRGBDFrame) -> bool:
        """Overwrite the single pending slot with the newest raw frame."""
        with self.frame_condition:
            scene = self.scene
            if scene is None:
                return False
            if self.fatal_error is not None:
                return False
            if scene.stop_event.is_set():
                return False
            self.pending_frame = (int(frame_id), frame)
            self.frame_condition.notify()
        return True

    def _frame_loop(self, scene: SceneRuntime) -> None:
        """Decode and process one selected latest frame at a time."""
        while True:
            with self.frame_condition:
                while self.pending_frame is None and not scene.stop_event.is_set():
                    self.frame_condition.wait()
                if self.pending_frame is None and scene.stop_event.is_set():
                    break
                frame_id, raw_frame = self.pending_frame
                self.pending_frame = None
                self.frame_active = True

            try:
                frame = self.decode_frame(raw_frame, scene.width, scene.height)
            except Exception as exc:
                logger.warning(
                    "Skipping invalid Redis frame timestamp={}: {}",
                    raw_frame.timestamp,
                    exc,
                )
            else:
                try:
                    self.process_frame(frame_id, frame, scene)
                except Exception as exc:
                    self._record_fatal(exc, scene)
            finally:
                with self.frame_condition:
                    self.frame_active = False
                    self.frame_condition.notify_all()

    def _record_fatal(self, error: Exception, scene: SceneRuntime) -> None:
        """Stop frame intake and notify the Engine about a fatal pipeline error."""
        scene.publish_gate.cancel()
        with self.frame_condition:
            if self.fatal_error is not None:
                return
            self.fatal_error = error
            self.pending_frame = None
            scene.stop_event.set()
            self.frame_condition.notify_all()
        if scene.fuse_worker is not None:
            scene.fuse_worker.request_abort()
        logger.opt(exception=error).error(
            "Realtime visualization processor failed: {}", error
        )
        if self.error_callback is not None:
            self.error_callback(error)

    def process_frame(
        self, frame_id: int, frame: RGBDFrame, scene: SceneRuntime
    ) -> None:
        """Run DINO, Boxer, visualization and incremental Fuse for one frame."""
        if scene.stop_event.is_set():
            return
        datum = self._build_datum(frame, scene)
        if scene.stop_event.is_set():
            return
        img_np = torch2cv2(datum["img0"], rotate=False, ensure_rgb=True)
        image_path = os.path.join(scene.temp_dir, f"{frame.timestamp}.jpg")
        try:
            with open(image_path, "wb") as image_file:
                image_file.write(frame.rgb_bytes)
            (
                bb2d,
                scores2d,
                labels2d,
                vis_bb2d,
                _,
                vis_labels2d,
                filtered_source_indices,
            ) = detect_groundingdino(
                self.groundingdino,
                datum,
                img_np,
                image_path,
                self.config["groundingdino"],
                self.boxernet,
                self.config["depth_filter_min_m"],
                self.config["depth_filter_max_m"],
            )
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

        if scene.stop_event.is_set():
            return

        obb_pr_w = datum["obbs"]
        labels3d = []
        vis_source_3d = torch.zeros(0, dtype=torch.long)
        if bb2d.shape[0] > 0:
            (
                obb_pr_w,
                scores3d,
                labels3d,
                bb2d_for_3d,
                source_2d_for_3d,
            ) = self._run_boxer(datum, bb2d, labels2d, scores2d)
            (
                obb_pr_w,
                _,
                labels3d,
                _,
                validation_keep,
            ) = filter_3dbb_by_validation(
                obb_pr_w,
                bb2d_for_3d,
                scores3d,
                labels3d,
                datum["sdp_w"].float(),
                datum["cam0"].float(),
                datum["T_world_rig0"].float(),
                img_np.shape[0],
                img_np.shape[1],
                self.boxernet.dino.patch_size,
                self.config["validation"],
                return_stats=True,
                return_keep=True,
            )
            source_2d_for_3d = source_2d_for_3d[validation_keep]
            vis_source_3d = filtered_source_indices[source_2d_for_3d]
            obb_pr_w.set_inst_id(source_2d_for_3d.to(dtype=torch.int32))

        if scene.stop_event.is_set():
            return

        published = write_detection_images(
            output_2d_path=scene.output_2d_dir / f"{frame.timestamp}.jpg",
            output_3d_path=scene.output_3d_dir / f"{frame.timestamp}.jpg",
            publish_gate=scene.publish_gate,
            img_np=img_np,
            orig_img_np=frame.rgb_bgr,
            bb2d=vis_bb2d,
            labels2d=vis_labels2d,
            obb_pr_w=obb_pr_w,
            labels3d=labels3d,
            T_wr=datum["T_world_rig0"].float(),
            cam=datum["cam0"].float(),
            boxernet=self.boxernet,
            frame_id=frame_id,
            time_ns=frame.timestamp,
            vocabulary=self.vocabulary,
            bb3_source_2d_indices=vis_source_3d,
            detection_line_thickness=self.config["visualization"][
                "detection_line_thickness"
            ],
            detection_label_font_size=self.config["visualization"][
                "detection_label_font_size"
            ],
            semantic_colors=True,
        )
        if not published:
            return
        logger.debug(
            "visualized frame={} timestamp={} 2dbb={} 3dbb={}",
            frame_id,
            frame.timestamp,
            len(vis_bb2d),
            len(obb_pr_w),
        )
        if (
            len(obb_pr_w) > 0
            and not scene.stop_event.is_set()
            and scene.fuse_worker is not None
        ):
            scene.fuse_worker.submit(
                FrameDetections(
                    obbs=obb_pr_w.clone().cpu(),
                    frame_id=frame_id,
                    time_ns=frame.timestamp,
                )
            )

    def request_stop(self) -> None:
        """Cancel publication and discard queued work for the active scene."""
        scene = self.scene
        if scene is None:
            return
        scene.publish_gate.cancel()
        with self.frame_condition:
            scene.stop_event.set()
            self.pending_frame = None
            self.frame_condition.notify_all()
        if scene.fuse_worker is not None:
            scene.fuse_worker.request_abort()

    def close(self) -> None:
        """Stop and join active scene workers while retaining resident models."""
        scene = self.scene
        if scene is None:
            return
        self.request_stop()
        if scene.frame_thread is not None:
            scene.frame_thread.join()
        if scene.fuse_worker is not None:
            scene.fuse_worker.abort()
        scene.frame_thread = None
        scene.fuse_worker = None
        shutil.rmtree(scene.temp_dir, ignore_errors=True)
        with self.frame_condition:
            if self.scene is scene:
                self.scene = None
                self.pending_frame = None
                self.frame_active = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def shutdown(self, timeout: float = 10.0) -> bool:
        """Release active scene workers and both resident models."""
        self.close()
        groundingdino_stopped = stop_groundingdino_server(
            self.groundingdino,
            timeout=max(float(timeout), 0.0),
        )
        if not groundingdino_stopped:
            return False
        self.groundingdino = None
        self.boxernet = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def force_stop_groundingdino(self) -> None:
        """Kill GroundingDINO immediately during forced shutdown."""
        process = self.groundingdino
        if process is not None and process.poll() is None:
            process.kill()

    def _build_datum(self, frame: RGBDFrame, scene: SceneRuntime) -> dict:
        """Build one Boxer-compatible in-memory datum."""
        fx = float(scene.intrinsics[0, 0])
        fy = float(scene.intrinsics[1, 1])
        cx = float(scene.intrinsics[0, 2])
        cy = float(scene.intrinsics[1, 2])
        params = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
        camera = CameraTW.from_surreal(
            width=scene.width,
            height=scene.height,
            type_str="pinhole",
            params=params,
        )
        pose = self._build_pose(frame.pose)
        points = self._sample_depth_points(
            frame.depth_m,
            fx,
            fy,
            cx,
            cy,
            pose.R.numpy().astype(np.float32),
            pose.t.numpy().astype(np.float32),
            self.config["num_samples"],
        )
        image = torch.from_numpy(frame.rgb).permute(2, 0, 1).float()[None] / 255.0
        resize_hw = int(self.boxernet.hw)
        image = torch.nn.functional.interpolate(
            image,
            size=(resize_hw, resize_hw),
            mode="bilinear",
            align_corners=True,
        )
        camera = camera.scale((resize_hw / scene.width, resize_hw / scene.height))
        return {
            "img0": image.float(),
            "cam0": camera.float(),
            "T_world_rig0": pose.float(),
            "sdp_w": points.float(),
            "time_ns0": frame.timestamp,
            "rotated0": torch.tensor(False).reshape(1),
            "obbs": ObbTW(torch.empty((0, 165), dtype=torch.float32)),
        }

    def _run_boxer(
        self,
        datum: dict,
        bb2d: torch.Tensor,
        labels2d: list[str],
        scores2d: torch.Tensor,
    ) -> tuple:
        """Run Boxer and retain confidence-qualified 3D boxes."""
        datum["bb2d"] = bb2d
        if self.device == "cuda":
            precision = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float32
            )
            with torch.autocast(device_type="cuda", dtype=precision):
                outputs = self.boxernet.forward(datum)
        else:
            outputs = self.boxernet.forward(datum)
        obb_pr_w = outputs["obbs_pr_w"].cpu()[0]

        semantic_ids = torch.zeros(len(labels2d), dtype=torch.int32)
        for index, label in enumerate(labels2d):
            if label not in self.sem_name_to_id:
                semantic_id = len(self.sem_name_to_id)
                self.sem_name_to_id[label] = semantic_id
                self.sem_id_to_name[semantic_id] = label
            semantic_ids[index] = self.sem_name_to_id[label]
        obb_pr_w.set_sem_id(semantic_ids)

        scores3d = obb_pr_w.prob.squeeze(-1).clone()
        keep = scores3d >= self.config["thresh3d"]
        source_indices = torch.arange(bb2d.shape[0], dtype=torch.long)
        bb2d_for_3d = bb2d[keep].clone()
        source_2d_for_3d = source_indices[keep].clone()
        obb_pr_w = obb_pr_w[keep].clone()
        scores3d = scores3d[keep].clone()
        labels3d = [
            labels2d[index] for index, keep_one in enumerate(keep.tolist()) if keep_one
        ]
        obb_pr_w.set_prob((scores2d[keep] + scores3d) / 2.0)
        if labels3d:
            text = torch.stack(
                [
                    string2tensor(pad_string(label, max_len=128))
                    for label in labels3d
                ]
            )
            obb_pr_w.set_text(text)
        return (
            obb_pr_w,
            scores3d,
            labels3d,
            bb2d_for_3d,
            source_2d_for_3d,
        )

    @staticmethod
    def _build_pose(
        pose: tuple[float, float, float, float, float, float, float]
    ) -> PoseTW:
        """Convert an xyzw pose to the aligned Boxer world frame."""
        tx, ty, tz, qx, qy, qz, qw = pose
        quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float32)
        quaternion /= np.linalg.norm(quaternion)
        qx, qy, qz, qw = quaternion
        rotation = np.asarray(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
            ],
            dtype=np.float32,
        )
        translation = np.asarray([tx, ty, tz], dtype=np.float32)
        rotation = R_ALIGN.astype(np.float32) @ rotation
        translation = R_ALIGN.astype(np.float32) @ translation
        return PoseTW.from_Rt(
            torch.from_numpy(rotation), torch.from_numpy(translation)
        )

    @staticmethod
    def _sample_depth_points(
        depth: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        rotation: np.ndarray,
        translation: np.ndarray,
        num_samples: int,
    ) -> torch.Tensor:
        """Sample semi-dense world points from one metric depth map."""
        height, width = depth.shape
        step = max(1, int(np.sqrt(height * width / (num_samples * 2))))
        yy, xx = np.mgrid[0:height:step, 0:width:step]
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)
        zz = depth[yy, xx]
        valid = zz > 0
        yy, xx, zz = yy[valid], xx[valid], zz[valid]
        if len(zz) > num_samples:
            indices = np.random.choice(len(zz), size=num_samples, replace=False)
            yy, xx, zz = yy[indices], xx[indices], zz[indices]
        if len(zz) == 0:
            return torch.zeros((0, 3), dtype=torch.float32)

        x3d = (xx.astype(np.float32) - cx) / fx * zz
        y3d = (yy.astype(np.float32) - cy) / fy * zz
        camera_points = np.stack([x3d, y3d, zz], axis=-1)
        world_points = camera_points @ rotation.T + translation
        result = torch.from_numpy(world_points.astype(np.float32))
        if result.shape[0] < num_samples:
            padding = torch.full(
                (num_samples - result.shape[0], 3),
                float("nan"),
                dtype=torch.float32,
            )
            result = torch.cat([result, padding], dim=0)
        return result.float()
