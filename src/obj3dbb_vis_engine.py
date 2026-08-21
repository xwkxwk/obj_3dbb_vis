"""Realtime_Processor_Engine adapter for obj-centric 3DBB visualization."""

import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from infrastructure.framework.engine import Realtime_Processor_Engine
from infrastructure.obj.config import Config
from infrastructure.repo.redis import Redis_Client
from pipeline.config import build_realtime_config
from realtime_vis_processor import RedisRGBDFrame, RealtimeVisProcessor


class Obj3DBBVis_Realtime_Engine(Realtime_Processor_Engine):
    """Provide obj3dbb hooks while the base Engine owns the lifecycle."""

    def __init__(
        self,
        redis_client: Redis_Client,
        processor_config: dict,
        full_config: dict,
        config_path: str,
    ) -> None:
        """Store algorithm configuration and initialize the base Engine."""
        self.full_config = full_config
        self.config_path = config_path
        self.rgb_topic = None
        self.intrinsics = None
        self.width = None
        self.height = None
        super().__init__(redis_client, processor_config)

    def register_camera_params_route(self) -> None:
        """Register the Splatam-style external camera parameter endpoint."""

        @self._app.route("/api/realtime/recon3d-params", methods=["POST"])
        def set_camera_params() -> tuple:
            """Store camera intrinsics until the service process exits."""
            data = self._http_request.get_json()
            camera = data["intrinsics"]
            self.rgb_topic = data["topic_name"]
            self.intrinsics = camera["intrinsics"]
            self.width = camera["width"]
            self.height = camera["height"]
            logger.info(
                "Camera parameters set: topic={} width={} height={}",
                self.rgb_topic,
                self.width,
                self.height,
            )
            return self._resp(msg="ok")

    def init_processor(self, **ext_config: Any) -> RealtimeVisProcessor:
        """Load resident models through the base load_processor lifecycle."""
        self.register_camera_params_route()
        run_config = build_realtime_config(self.full_config, self.config_path)
        processor = RealtimeVisProcessor(run_config)
        logger.info(
            "Obj3DBB visualization models loaded: gpu_mode={} "
            "detection_device={} fuse_device={}",
            run_config["gpu_mode"],
            run_config["detection_device"],
            run_config["fuse_device"],
        )
        return processor

    def pre_start_thread(self, scene_id: str) -> None:
        """Initialize scene output and internal latest-frame workers."""
        self.output_path = self.scene_path / "obj3dbb_vis"
        self.processor.pre_start(
            self.output_path,
            self.intrinsics,
            self.width,
            self.height,
            self._handle_processor_error,
        )
        logger.info(
            "Obj3DBB scene started: scene_id={} topic={} output={}",
            scene_id,
            self.input_topic_name,
            self.output_path,
        )

    def handle_inputdata(
        self, message: dict
    ) -> tuple[int, Optional[RedisRGBDFrame]]:
        """Extract one raw RGB-D frame in Redis Stream order."""
        try:
            for key in ("ts", "content", "depth", "camera_pose"):
                if key not in message:
                    raise ValueError(f"missing Redis field: {key}")
            raw_timestamp = message["ts"]
            if isinstance(raw_timestamp, bytes):
                raw_timestamp = raw_timestamp.decode("utf-8")
            timestamp = int(raw_timestamp)
            if timestamp < 0:
                raise ValueError("timestamp must be non-negative")
            frame = RedisRGBDFrame(
                timestamp=timestamp,
                rgb_bytes=message["content"],
                depth_bytes=message["depth"],
                camera_pose=message["camera_pose"],
            )
            return timestamp, frame
        except Exception as exc:
            logger.warning("Skipping invalid Redis frame: {}", exc)
            return 0, None

    def do_processor_function(
        self, timestamp: int, data: dict
    ) -> tuple[int, None]:
        """Submit the frame without blocking the base Redis reader thread."""
        frame_id = max(int(self.frame_no) - 1, 0)
        self.processor.submit_frame(frame_id, data["input_data"])
        return timestamp, None

    def get_result(self, timestamp: int) -> None:
        """Return no in-memory result because files are the output contract."""
        return None

    def post_handle_processor(self, timestamp: int, data: Any) -> None:
        """Publish no Redis result."""
        return None

    def stop_processor(self) -> None:
        """Stop current scene workers while retaining models and service."""
        self.processor.close()
        logger.info("Obj3DBB scene processor stopped; resident models retained")

    def _handle_processor_error(self, error: Exception) -> None:
        """Stop the base Redis loop after an asynchronous processor failure."""
        self.stop_flag = True
        logger.opt(exception=error).error(
            "Obj3DBB asynchronous processing stopped: {}", error
        )

def setup_logger(project_root: Path, mode: str) -> None:
    """Configure console and hourly compressed file logs."""
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    level = "DEBUG" if str(mode).lower() == "debug" else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True)
    logger.add(
        str(log_dir / "obj3dbb_vis_{time:YYYY_MM_DD_HH}.log"),
        level=level,
        rotation="1 hour",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
    )


def init() -> tuple[Redis_Client, dict, dict, str]:
    """Load configuration and construct dependencies for the base Engine."""
    configuration = Config()
    full_config = configuration.orig_config
    config_path = str(Path(configuration.config_path).resolve())
    project_root = Path(__file__).resolve().parents[1]
    setup_logger(project_root, (full_config.get("log") or {}).get("mode", "release"))
    redis_config = full_config["redis"]
    redis_client = Redis_Client(
        host=redis_config["host"],
        port=redis_config["port"],
        db=redis_config.get("db", 0),
        password=redis_config.get("password"),
        encoding="UTF-8",
    )
    processor_config = dict(full_config["service"])
    if processor_config.get("base_data_path") is None:
        processor_config["base_data_path"] = full_config["system"]["base_data_path"]
    return redis_client, processor_config, full_config, config_path


def main() -> None:
    """Start the processor through the base load and run lifecycle."""
    redis_client, processor_config, full_config, config_path = init()
    engine = Obj3DBBVis_Realtime_Engine(
        redis_client,
        processor_config,
        full_config,
        config_path,
    )
    engine.load_processor()
    engine.run()


if __name__ == "__main__":
    main()
