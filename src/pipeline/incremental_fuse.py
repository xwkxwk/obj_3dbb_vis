"""Incremental Fuse state and background snapshot worker"""

import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from loguru import logger

from pipeline.output_writer import (
    ScenePublishGate,
    pipeline_log,
    write_sfm_fused_csv,
)
from utils.file_io import ObbCsvWriter2
from utils.fuse_3d_boxes import (
    BoundingBox3DFuser,
    FusedInstance,
    apply_nms_to_fused_instances,
    apply_same_semantic_ios_absorb,
    build_one_hot_semantic_embeddings,
    rename_fused_instance_column,
)
from utils.tw.obb import ObbTW


@dataclass
class FrameDetections:
    """Final 3DBB detections from one input frame"""

    obbs: ObbTW
    frame_id: int
    time_ns: int


@dataclass
class DetectionRecord:
    """One persistent raw 3DBB member"""

    obb: ObbTW
    frame_id: int
    time_ns: int
    instance: int


@dataclass
class IncrementalComponent:
    """One tentative or publishable fused component"""

    obb: ObbTW
    member_ids: list[int]
    stable_id: int


class IncrementalFuser:
    """Fuse new detections with spatially affected persistent components"""

    def __init__(self, fuse_cfg: dict, device: str) -> None:
        """Initialize CPU-persistent Fuse records and stable IDs."""
        self.fuse_cfg = fuse_cfg
        self.device = device
        self.records: list[DetectionRecord] = []
        self.components: list[IncrementalComponent] = []
        self.aliases: dict[int, int] = {}
        self.next_stable_id = 0

    def update(self, frames: tuple[FrameDetections, ...]) -> None:
        """Apply one immutable frame batch to current Fuse state"""
        new_ids = self._append_records(frames)
        if len(new_ids) == 0:
            return

        old_components = self.components
        affected = self._affected_components(new_ids)
        local_ids = list(new_ids)
        for index in affected:
            local_ids.extend(old_components[index].member_ids)
        local_ids = list(dict.fromkeys(local_ids))

        local_instances = self._fuse_local(local_ids)
        unaffected_instances = [
            self._instance_from_component(component)
            for index, component in enumerate(old_components)
            if index not in affected
        ]
        all_instances = unaffected_instances + local_instances
        all_instances = self._postprocess(all_instances)
        self.components = self._assign_stable_ids(all_instances, old_components)

    def published_components(self) -> list[IncrementalComponent]:
        """Return components meeting the configured support threshold"""
        minimum = int(self.fuse_cfg["min_detections"])
        return sorted(
            (
                component
                for component in self.components
                if len(component.member_ids) >= minimum
            ),
            key=lambda component: component.stable_id,
        )

    def _append_records(self, frames: tuple[FrameDetections, ...]) -> list[int]:
        """Append confidence-qualified raw detections"""
        new_ids = []
        threshold = float(self.fuse_cfg["conf_threshold"])
        for frame in frames:
            for obb in frame.obbs:
                probability = float(obb.prob.reshape(-1)[0].item())
                if probability < threshold:
                    continue
                record = DetectionRecord(
                    obb=obb.clone().cpu(),
                    frame_id=int(frame.frame_id),
                    time_ns=int(frame.time_ns),
                    instance=int(obb.inst_id.reshape(-1)[0].item()),
                )
                self.records.append(record)
                new_ids.append(len(self.records) - 1)
        return new_ids

    def _affected_components(self, new_ids: list[int]) -> set[int]:
        """Find old components whose member bounds intersect new boxes"""
        if len(self.components) == 0:
            return set()
        new_bounds = [self._record_bounds(record_id) for record_id in new_ids]
        affected = set()
        for component_index, component in enumerate(self.components):
            component_min, component_max = self._component_bounds(component)
            if any(
                np.all(component_max >= box_min) and np.all(box_max >= component_min)
                for box_min, box_max in new_bounds
            ):
                affected.add(component_index)
        return affected

    def _record_bounds(self, record_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the axis-aligned world bounds of one raw detection."""
        corners = (
            self.records[record_id]
            .obb.bb3corners_world.detach()
            .cpu()
            .numpy()
            .reshape(-1, 3)
        )
        return corners.min(axis=0), corners.max(axis=0)

    def _component_bounds(
        self, component: IncrementalComponent
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return aggregate world bounds for one incremental component."""
        bounds = [self._record_bounds(record_id) for record_id in component.member_ids]
        minimum = np.min(np.stack([item[0] for item in bounds]), axis=0)
        maximum = np.max(np.stack([item[1] for item in bounds]), axis=0)
        return minimum, maximum

    def _fuse_local(self, member_ids: list[int]) -> list[FusedInstance]:
        """Run existing robust clustering on one affected region"""
        if len(member_ids) == 0:
            return []
        detections = torch.stack([self.records[index].obb for index in member_ids])
        frame_ids = torch.tensor(
            [self.records[index].frame_id for index in member_ids],
            dtype=torch.long,
        )
        embeddings = build_one_hot_semantic_embeddings(detections)
        local_cfg = self.fuse_cfg
        fuser = BoundingBox3DFuser(
            cluster_iou_threshold=float(local_cfg["cluster_iou_threshold"]),
            min_detections=1,
            conf_threshold=0.0,
            semantic_threshold=float(local_cfg["semantic_threshold"]),
            enable_nms=False,
            nms_iou_threshold=float(local_cfg["nms_iou"]),
            same_semantic_ios_absorb=False,
            ios_absorb_threshold=float(local_cfg["ios_absorb_threshold"]),
            device=self.device,
        )
        instances = fuser.fuse(
            detections,
            semantic_embeddings=embeddings,
            frame_ids=frame_ids,
        )
        for instance in instances:
            instance.detection_indices = [
                member_ids[local_index]
                for local_index in instance.detection_indices
            ]
            instance.support_count = len(instance.detection_indices)
            instance.source_members = self._sources(instance.detection_indices)
        return instances

    def _postprocess(self, instances: list[FusedInstance]) -> list[FusedInstance]:
        """Apply configured postprocessing to publishable instances"""
        minimum = int(self.fuse_cfg["min_detections"])
        tentative = [item for item in instances if item.support_count < minimum]
        confirmed = [item for item in instances if item.support_count >= minimum]
        if self.fuse_cfg["nms"]:
            confirmed = apply_nms_to_fused_instances(
                confirmed,
                float(self.fuse_cfg["nms_iou"]),
                device=self.device,
            )
        if self.fuse_cfg["same_semantic_ios_absorb"]:
            confirmed = apply_same_semantic_ios_absorb(
                confirmed,
                float(self.fuse_cfg["ios_absorb_threshold"]),
                device=self.device,
            )
        return tentative + confirmed

    def _assign_stable_ids(
        self,
        instances: list[FusedInstance],
        old_components: list[IncrementalComponent],
    ) -> list[IncrementalComponent]:
        """Preserve IDs across component merge and split operations"""
        instance_members = [set(item.detection_indices) for item in instances]
        claims: list[list[int]] = [[] for _ in instances]
        for old_component in old_components:
            old_members = set(old_component.member_ids)
            overlaps = [len(old_members & members) for members in instance_members]
            if len(overlaps) == 0 or max(overlaps) == 0:
                continue
            winner = max(range(len(overlaps)), key=lambda index: overlaps[index])
            claims[winner].append(old_component.stable_id)

        components = []
        for index, instance in enumerate(instances):
            inherited = sorted(set(claims[index]))
            if inherited:
                stable_id = inherited[0]
                for alias in inherited[1:]:
                    self.aliases[alias] = stable_id
            else:
                stable_id = self.next_stable_id
                self.next_stable_id += 1
            components.append(
                IncrementalComponent(
                    obb=instance.obb.clone().cpu(),
                    member_ids=list(instance.detection_indices),
                    stable_id=int(stable_id),
                )
            )
        return components

    def _instance_from_component(
        self, component: IncrementalComponent
    ) -> FusedInstance:
        """Convert persistent component state to the original Fuse structure."""
        return FusedInstance(
            obb=component.obb.clone().cpu(),
            support_count=len(component.member_ids),
            detection_indices=list(component.member_ids),
            source_members=self._sources(component.member_ids),
        )

    def _sources(self, member_ids: list[int]) -> list[dict]:
        """Return the persistent source-frame metadata for component members."""
        return [
            {
                "time_ns": self.records[index].time_ns,
                "frame_id": self.records[index].frame_id,
                "instance": self.records[index].instance,
            }
            for index in member_ids
        ]


class IncrementalFuseWorker:
    """Fuse detections serially and atomically publish realtime snapshots."""

    def __init__(
        self,
        fuse_cfg: dict,
        device: str,
        output_dir: str,
        vocabulary: dict,
        publish_gate: ScenePublishGate,
        log_callback: Optional[Callable[[str], None]],
        error_callback: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """Start the one-at-a-time realtime Fuse background thread."""
        self.fuser = IncrementalFuser(fuse_cfg, device)
        self.device = torch.device(device)
        self.cuda_enabled = self.device.type == "cuda" and torch.cuda.is_available()
        self.output_dir = output_dir
        self.vocabulary = vocabulary
        self.publish_gate = publish_gate
        self.log_callback = log_callback
        self.error_callback = error_callback
        self.condition = threading.Condition()
        self.next_batch = None
        self.pending_frames: list[FrameDetections] = []
        self.running = False
        self.closing = False
        self.error = None
        self.snapshot_count = 0
        self.last_timestamp = None
        self.last_fused_count = 0
        self.aborted = False
        self.thread = threading.Thread(
            target=self._run,
            name="obj3dbb-fuse",
            daemon=True,
        )
        self.thread.start()

    def submit(self, frame: FrameDetections) -> None:
        """Start Fuse immediately or accumulate the frame for the next Fuse"""
        with self.condition:
            if self.error is not None:
                raise RuntimeError(
                    f"incremental Fuse failed: {self.error}"
                ) from self.error
            if self.closing or self.aborted:
                raise RuntimeError("incremental Fuse worker is not accepting frames")
            if self.next_batch is None and not self.running:
                self.next_batch = (frame,)
            else:
                self.pending_frames.append(frame)
            self.condition.notify()

    def request_abort(self) -> None:
        """Discard queued Fuse work without waiting for the active batch."""
        self.publish_gate.cancel()
        with self.condition:
            self.aborted = True
            self.next_batch = None
            self.pending_frames.clear()
            self.condition.notify()

    def abort(self, timeout: Optional[float] = None) -> bool:
        """Discard queued work and wait up to timeout for the active Fuse."""
        self.request_abort()
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    def raise_if_failed(self) -> None:
        """Raise the stored Fuse worker failure, if any."""
        if self.error is not None:
            raise RuntimeError(f"incremental Fuse failed: {self.error}") from self.error

    def _run(self) -> None:
        """Process one active batch and one accumulated next batch at a time."""
        while True:
            with self.condition:
                while self.next_batch is None and not self.aborted:
                    self.condition.wait()
                if self.aborted:
                    break
                frames = self.next_batch
                self.next_batch = None
                self.running = True

            cutoff_time_ns = int(frames[-1].time_ns)
            try:
                self.fuser.update(frames)
                if self.publish_gate.is_cancelled():
                    count, published = 0, False
                else:
                    count, published = self._write_snapshot(cutoff_time_ns)
                if published:
                    self.snapshot_count += 1
                    self.last_timestamp = cutoff_time_ns
                    self.last_fused_count = count
                    pipeline_log(
                        self.log_callback,
                        f"incremental_fuse timestamp={cutoff_time_ns} "
                        f"frames={len(frames)} fused_3dbb={count}",
                    )
            except Exception as exc:
                self._record_error(exc, cutoff_time_ns)
            finally:
                if self.cuda_enabled:
                    try:
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()
                    except Exception as exc:
                        self._record_error(exc, cutoff_time_ns)
            with self.condition:
                self.running = False
                if self.error is None and not self.aborted and self.pending_frames:
                    self.next_batch = tuple(self.pending_frames)
                    self.pending_frames.clear()
                elif self.error is not None:
                    self.pending_frames.clear()
                    self.closing = True
                self.condition.notify_all()

    def _record_error(self, error: Exception, cutoff_time_ns: int) -> None:
        """Store one fatal Fuse error and notify the processor."""
        if self.error is not None:
            return
        self.error = error
        logger.error("Incremental Fuse failed at {}: {}", cutoff_time_ns, error)
        if self.error_callback is not None:
            try:
                self.error_callback(error)
            except Exception:
                logger.exception("Incremental Fuse error callback failed")

    def _write_snapshot(self, cutoff_time_ns: int) -> tuple[int, bool]:
        """Write one complete SFM snapshot and publish atomically"""
        components = self.fuser.published_components()
        internal_path = os.path.join(
            self.output_dir, f".{cutoff_time_ns}.internal.csv"
        )
        sfm_path = os.path.splitext(internal_path)[0] + "_sfm.csv"
        target_path = os.path.join(self.output_dir, f"{cutoff_time_ns}.csv")
        try:
            writer = ObbCsvWriter2(internal_path)
            if components:
                obbs = torch.stack([component.obb for component in components])
                stable_ids = torch.tensor(
                    [component.stable_id for component in components],
                    dtype=torch.int32,
                )
                obbs.set_inst_id(stable_ids)
                probabilities = torch.round(obbs.prob * 100) / 100
                obbs.set_prob(probabilities.squeeze(-1), use_mask=False)
                names = obbs.text_string()
                semantic_ids = obbs.sem_id.squeeze(-1).cpu().tolist()
                sem_id_to_name = {
                    int(semantic_id): names[index]
                    for index, semantic_id in enumerate(semantic_ids)
                }
                writer.write(
                    obbs,
                    timestamps_ns=cutoff_time_ns,
                    sem_id_to_name=sem_id_to_name,
                )
            writer.close()
            rename_fused_instance_column(internal_path)
            write_sfm_fused_csv(
                internal_path,
                self.vocabulary,
                semantic_colors=True,
            )
            published = self.publish_gate.publish(
                lambda: os.replace(sfm_path, target_path)
            )
        finally:
            for path in (internal_path, sfm_path):
                if os.path.exists(path):
                    os.remove(path)
        return len(components), published
