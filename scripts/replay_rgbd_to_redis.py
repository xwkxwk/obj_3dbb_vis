"""Replay one offline RGB-D sequence into a Redis Stream."""

import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import redis


def load_poses(path: Path) -> dict[str, tuple[float, ...]]:
    """Load timestamped tx ty tz qx qy qz qw camera poses."""
    poses = {}
    with path.open("r", encoding="utf-8") as pose_file:
        for line in pose_file:
            parts = line.split()
            if len(parts) != 8:
                continue
            poses[parts[0]] = tuple(float(value) for value in parts[1:])
    return poses


def index_images(path: Path) -> dict[str, Path]:
    """Index supported image files by timestamp stem."""
    return {
        image_path.stem: image_path
        for image_path in path.iterdir()
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


def encode_rgb(path: Path) -> bytes:
    """Read one RGB image and return JPEG bytes."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read RGB image: {path}")
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not success:
        raise ValueError(f"cannot encode RGB image: {path}")
    return encoded.tobytes()


def encode_depth(path: Path) -> bytes:
    """Read millimetre uint16 depth and return PNG bytes."""
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"cannot read depth image: {path}")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(
            f"depth must be single-channel uint16: {path}, "
            f"shape={depth.shape}, dtype={depth.dtype}"
        )
    success, encoded = cv2.imencode(".png", depth)
    if not success:
        raise ValueError(f"cannot encode depth image: {path}")
    return encoded.tobytes()


def encode_pose(values: tuple[float, ...]) -> bytes:
    """Encode one xyzw camera pose using the service pickle schema."""
    tx, ty, tz, qx, qy, qz, qw = values
    pose = {
        "position": {"x": tx, "y": ty, "z": tz},
        "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
    }
    return pickle.dumps(pose, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:
    """Parse replay and Redis connection arguments."""
    parser = argparse.ArgumentParser(
        description="Replay RGB-D frames into a Redis Stream."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--stream",
        default="simu4d:perceive:camera",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--db", type=int, default=0)
    parser.add_argument("--password", default="123456")
    parser.add_argument("--clear-stream", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Publish matched RGB, depth and pose frames at a fixed rate."""
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be greater than zero")

    rgb_paths = index_images(args.data / "images")
    depth_paths = index_images(args.data / "depth")
    poses = load_poses(args.data / "poses.txt")
    timestamps = [
        timestamp
        for timestamp in poses
        if timestamp in rgb_paths and timestamp in depth_paths
    ]
    if not timestamps:
        raise ValueError("no matching RGB, depth and pose timestamps")

    client = redis.Redis(
        host=args.host,
        port=args.port,
        db=args.db,
        password=args.password or None,
    )
    client.ping()
    if args.clear_stream:
        client.delete(args.stream)

    period = 1.0 / args.fps
    deadline = time.monotonic()
    for index, timestamp in enumerate(timestamps, start=1):
        client.xadd(
            args.stream,
            {
                "ts": timestamp,
                "content": encode_rgb(rgb_paths[timestamp]),
                "depth": encode_depth(depth_paths[timestamp]),
                "camera_pose": encode_pose(poses[timestamp]),
            },
            maxlen=1024,
            approximate=True,
        )
        print(
            f"[{index}/{len(timestamps)}] "
            f"stream={args.stream} ts={timestamp}",
            flush=True,
        )
        deadline += period
        if index < len(timestamps):
            time.sleep(max(0.0, deadline - time.monotonic()))

    print(f"Replay completed: {len(timestamps)} frames", flush=True)


if __name__ == "__main__":
    main()
