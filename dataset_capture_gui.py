#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ROS_SETUP = Path("/opt/ros/humble/setup.bash")
DEFAULT_OUTPUT_ROOT = "/home/xyz-ai/baris_vision_data_capture/datasets"
DEFAULT_COLOR_TOPIC = "/camera/color/image_raw"
DEFAULT_SERVING_TOPIC = "/latest_serving_decision"
DEFAULT_YOLO_MODEL_PATH = (
    "/home/xyz-ai/VisionXOnJetson/src/smart_pickupzone/"
    "smart_pickupzone/model/tensorrt/yolo26s.engine"
)


def ensure_ros_humble_environment() -> None:
    if os.environ.get("ROS_DISTRO") == "humble" and "/opt/ros/humble" in os.environ.get(
        "AMENT_PREFIX_PATH", ""
    ):
        return
    if os.environ.get("_DATASET_CAPTURE_ROS_BOOTSTRAPPED") == "1":
        return
    if not ROS_SETUP.exists():
        raise RuntimeError(f"ROS2 Humble setup file not found: {ROS_SETUP}")

    env = os.environ.copy()
    env["_DATASET_CAPTURE_ROS_BOOTSTRAPPED"] = "1"
    argv = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    command = f"source {shlex.quote(str(ROS_SETUP))} && exec {shlex.join(argv)}"
    os.execvpe("bash", ["bash", "-lc", command], env)


ensure_ros_humble_environment()

import cv2  # noqa: E402
import rclpy  # noqa: E402
import tkinter as tk  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402


StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class CaptureConfig:
    output_root: Path
    site_code: str
    camera_kind: str
    dataset_type: str
    max_events: int
    duration_sec: float
    target_fps: float
    serving_topic: str
    color_topic: str
    yolo_model_path: str
    conf_threshold: float
    target_label: str
    skip_yolo: bool

    @property
    def max_frames(self) -> int:
        return int(round(self.duration_sec * self.target_fps))


@dataclass(frozen=True)
class ServingEvent:
    timestamp: str
    serving_decision: int
    order_num: str | None
    order_count: int | None


@dataclass(frozen=True)
class Detection:
    class_id: int | None
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


class CaptureSession:
    def __init__(self, event: ServingEvent, config: CaptureConfig, now_monotonic: float):
        self.event = event
        self.config = config
        self.started_monotonic = now_monotonic
        self.frame_count = 0
        self.finished_capture = False
        self.finalized = False
        self.recorded_width = 0
        self.recorded_height = 0
        self.dataset_timestamp = dataset_timestamp_for_path(event.timestamp)
        self.frame_timestamp = frame_timestamp_for_path(event.timestamp)
        self.dataset_id = build_dataset_id(config, self.dataset_timestamp)
        self.frame_prefix = build_file_prefix(config, self.frame_timestamp)
        self.label_prefix = self.frame_prefix
        self.event_dir = unique_event_dir(config.output_root, self.dataset_id)
        self.images_dir = self.event_dir / "images"
        self.bbox_dir = self.event_dir / "bbox"
        self.manifest_path = self.event_dir / "frame_manifest.jsonl"
        self.metadata_path = self.event_dir / "metadata.json"
        self.event_dir.mkdir(parents=True, exist_ok=False)
        self.images_dir.mkdir()
        self.bbox_dir.mkdir()
        write_metadata(self)

    def due_for_frame(self, now_monotonic: float) -> bool:
        if self.finished_capture:
            return False
        due_time = self.started_monotonic + (self.frame_count / self.config.target_fps)
        return now_monotonic + 1e-6 >= due_time

    def should_finish(self, now_monotonic: float) -> bool:
        elapsed = now_monotonic - self.started_monotonic
        return elapsed >= self.config.duration_sec or self.frame_count >= self.config.max_frames

    def frame_name(self, frame_index: int) -> str:
        return f"{self.frame_prefix}_frame_{frame_index:08d}.png"

    def label_name(self, frame_index: int) -> str:
        return f"{self.label_prefix}_frame_{frame_index:08d}.json"

    def frame_path(self, frame_index: int | None = None) -> Path:
        index = self.frame_count if frame_index is None else frame_index
        return self.images_dir / self.frame_name(index)

    def label_path(self, frame_index: int) -> Path:
        return self.bbox_dir / self.label_name(frame_index)

    def observe_frame_size(self, width: int, height: int) -> None:
        if self.frame_count == 0:
            self.recorded_width = width
            self.recorded_height = height


class GuiServingDatasetCapture(Node):
    def __init__(self, config: CaptureConfig, status: StatusCallback):
        super().__init__("gui_serving_dataset_capture")
        self.bridge = CvBridge()
        self.config = config
        self.status = status
        self.lock = threading.Lock()
        self.active_sessions: list[CaptureSession] = []
        self.finalizer_threads: list[threading.Thread] = []
        self.last_event_timestamp: str | None = None
        self.last_error: str | None = None
        self.started_event_count = 0
        self.completed_event_count = 0
        self.stop_requested = False

        qos_profile = QoSProfile(depth=10)
        self.create_subscription(
            String,
            config.serving_topic,
            self.serving_decision_callback,
            qos_profile,
        )
        self.create_subscription(
            Image,
            config.color_topic,
            self.color_callback,
            qos_profile,
        )
        self.create_timer(0.5, self.shutdown_if_done)

        self.report(
            "Capture ready: "
            f"output_root={config.output_root}, events={config.max_events}, "
            f"fps={config.target_fps}, duration={config.duration_sec}s"
        )

    def report(self, message: str) -> None:
        self.get_logger().info(message)
        self.status(message)

    def serving_decision_callback(self, msg: String) -> None:
        if self.stop_requested or self.started_event_count >= self.config.max_events:
            return

        try:
            event = parse_serving_event(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self.last_error = f"invalid serving event payload: {error}"
            self.get_logger().warning(self.last_error)
            self.status(self.last_error)
            return

        if event is None or event.timestamp == self.last_event_timestamp:
            return

        try:
            session = CaptureSession(
                event=event,
                config=self.config,
                now_monotonic=time.monotonic(),
            )
        except Exception as error:
            self.last_error = f"failed to create capture session: {error}"
            self.get_logger().error(self.last_error)
            self.status(self.last_error)
            return

        with self.lock:
            self.last_event_timestamp = event.timestamp
            self.started_event_count += 1
            self.active_sessions.append(session)
        self.report(
            f"Started event {self.started_event_count}/{self.config.max_events}: "
            f"{session.event_dir}"
        )

    def color_callback(self, msg: Image) -> None:
        now_monotonic = time.monotonic()
        with self.lock:
            sessions = list(self.active_sessions)
        if not sessions:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as error:
            self.last_error = f"failed to convert color image: {error}"
            self.get_logger().error(self.last_error)
            self.status(self.last_error)
            return

        height, width = frame.shape[:2]
        finished_sessions = []
        for session in sessions:
            if session.due_for_frame(now_monotonic):
                self._save_frame(session, frame, msg, now_monotonic, width, height)
            if session.should_finish(now_monotonic):
                session.finished_capture = True
                finished_sessions.append(session)

        if not finished_sessions:
            return

        with self.lock:
            self.active_sessions = [
                session for session in self.active_sessions if not session.finished_capture
            ]
        for session in finished_sessions:
            self._start_finalize_thread(session)

    def _save_frame(
        self,
        session: CaptureSession,
        frame: Any,
        msg: Image,
        now_monotonic: float,
        width: int,
        height: int,
    ) -> None:
        session.observe_frame_size(width, height)
        image_path = session.frame_path()
        if not cv2.imwrite(str(image_path), frame):
            self.last_error = f"failed to write frame {image_path}"
            self.get_logger().error(self.last_error)
            self.status(self.last_error)
            return

        ros_stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
        capture_time_sec = session.frame_count / session.config.target_fps
        append_frame_manifest(session, image_path, ros_stamp_sec, capture_time_sec)
        session.frame_count += 1
        write_metadata(session)

        if session.should_finish(now_monotonic):
            self.report(
                f"Finished frame capture: {session.event_dir} frames={session.frame_count}"
            )

    def _start_finalize_thread(self, session: CaptureSession) -> None:
        thread = threading.Thread(
            target=self._finalize_session,
            args=(session,),
            name=f"finalize-{session.dataset_id}",
            daemon=True,
        )
        self.finalizer_threads.append(thread)
        thread.start()

    def _finalize_session(self, session: CaptureSession) -> None:
        try:
            if session.frame_count == 0:
                raise RuntimeError("no frames captured")
            detections_by_frame = (
                empty_detections(session)
                if session.config.skip_yolo
                else self._run_yolo_batch(session)
            )
            write_annotation_files(session, detections_by_frame)
            session.finalized = True
            write_metadata(session)
            self.report(f"Finalized: {session.event_dir} frames={session.frame_count}")
        except Exception as error:
            self.last_error = f"failed to finalize {session.event_dir}: {error}"
            self.get_logger().error(self.last_error)
            self.status(self.last_error)
        finally:
            with self.lock:
                self.completed_event_count += 1

    def _run_yolo_batch(self, session: CaptureSession) -> dict[int, list[Detection]]:
        from ultralytics import YOLO

        self.status(f"Running YOLO bbox: {session.event_dir.name}")
        model = YOLO(session.config.yolo_model_path, task="detect")
        detections_by_frame: dict[int, list[Detection]] = {}
        for frame_index in range(session.frame_count):
            results = model.predict(
                str(session.frame_path(frame_index)),
                conf=session.config.conf_threshold,
                verbose=False,
            )
            detections_by_frame[frame_index] = (
                normalize_ultralytics_result(results[0], session.config.target_label)
                if results
                else []
            )
        return detections_by_frame

    def shutdown_if_done(self) -> None:
        with self.lock:
            done = (
                self.started_event_count >= self.config.max_events
                and self.completed_event_count >= self.config.max_events
                and not self.active_sessions
            )
        if done and not self.stop_requested:
            self.stop_requested = True
            self.report("Requested event count completed; shutting down ROS capture")
            if rclpy.ok():
                rclpy.shutdown()

    def request_stop(self) -> None:
        self.stop_requested = True
        self.status("Stop requested")
        if rclpy.ok():
            rclpy.shutdown()


def parse_serving_event(payload: str) -> ServingEvent | None:
    data = json.loads(payload)
    timestamp = data.get("timestamp")
    serving_decision = data.get("serving_decision")
    if not timestamp or serving_decision is None or int(serving_decision) == -1:
        return None
    return ServingEvent(
        timestamp=str(timestamp),
        serving_decision=int(serving_decision),
        order_num=optional_str(data.get("order_num")),
        order_count=optional_int(data.get("order_count")),
    )


def optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def sanitize_component(value: Any, default: str = "unknown") -> str:
    text = default if value is None or str(value) == "" else str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or default


def parse_event_datetime(timestamp: str) -> datetime:
    cleaned = timestamp.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return datetime.now().astimezone()
    if parsed.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        parsed = parsed.replace(tzinfo=local_tz)
    return parsed.astimezone()


def dataset_timestamp_for_path(timestamp: str) -> str:
    return parse_event_datetime(timestamp).strftime("%Y%m%dT%H%M%S")


def frame_timestamp_for_path(timestamp: str) -> str:
    return parse_event_datetime(timestamp).strftime("%Y%m%dT%H%M%S%z")


def build_dataset_id(config: CaptureConfig, dataset_timestamp: str) -> str:
    return "_".join(
        [
            sanitize_component(config.site_code, "unknown_site"),
            dataset_timestamp,
            sanitize_component(config.camera_kind, "unknown_camera"),
            sanitize_component(config.dataset_type, "unknown_type"),
        ]
    )


def build_file_prefix(config: CaptureConfig, timestamp: str) -> str:
    return "_".join(
        [
            sanitize_component(config.site_code, "unknown_site"),
            timestamp,
            sanitize_component(config.camera_kind, "unknown_camera"),
            sanitize_component(config.dataset_type, "unknown_type"),
        ]
    )


def unique_event_dir(output_root: Path, dataset_id: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    candidate = output_root / dataset_id
    if not candidate.exists():
        return candidate
    for suffix in range(1, 10000):
        candidate = output_root / f"{dataset_id}_{suffix:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"unable to allocate unique dataset directory for {dataset_id}")


def write_metadata(session: CaptureSession) -> None:
    payload = {
        "dataset_id": session.dataset_id,
        "finalized": session.finalized,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "naming": {
            "dataset_dir": "<site_code>_<timestamp_seconds>_<camera_kind>_<type>",
            "frame_file": (
                "<site_code>_<timestamp_seconds_with_timezone>_<camera_kind>_"
                "<type>_frame_00000000.png"
            ),
            "label_file": (
                "<site_code>_<timestamp_seconds_with_timezone>_<camera_kind>_"
                "<type>_frame_00000000.json"
            ),
            "site_code": session.config.site_code,
            "camera_kind": session.config.camera_kind,
            "type": session.config.dataset_type,
            "dataset_timestamp": session.dataset_timestamp,
            "frame_timestamp": session.frame_timestamp,
        },
        "trigger": {
            "topic": session.config.serving_topic,
            "timestamp": session.event.timestamp,
            "serving_decision": session.event.serving_decision,
            "order_num": session.event.order_num,
            "order_count": session.event.order_count,
        },
        "recording": {
            "duration_sec": session.config.duration_sec,
            "fps": session.config.target_fps,
            "frame_count": session.frame_count,
            "color_topic": session.config.color_topic,
            "width": session.recorded_width,
            "height": session.recorded_height,
            "images_dir": "images",
            "bbox_dir": "bbox",
            "frame_manifest": "frame_manifest.jsonl",
        },
        "annotation": {
            "model": session.config.yolo_model_path,
            "labels": [session.config.target_label],
            "skip_yolo": session.config.skip_yolo,
        },
    }
    session.metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_frame_manifest(
    session: CaptureSession,
    image_file: Path,
    ros_stamp_sec: float,
    capture_time_sec: float,
) -> None:
    frame_index = session.frame_count
    label_path = session.label_path(frame_index)
    row = {
        "frame_index": frame_index,
        "image_file": str(image_file.relative_to(session.event_dir)),
        "label_file": str(label_path.relative_to(session.event_dir)),
        "ros_stamp_sec": ros_stamp_sec,
        "capture_time_sec": capture_time_sec,
    }
    with session.manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_labelme_payload(
    image_path: str,
    width: int,
    height: int,
    detections: Iterable[Detection],
) -> dict[str, Any]:
    shapes = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        shapes.append(
            {
                "label": detection.class_name,
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": f"conf={detection.confidence:.4f}",
                "shape_type": "rectangle",
                "flags": {},
            }
        )
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def normalize_ultralytics_result(result: Any, target_label: str) -> list[Detection]:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    normalized: list[Detection] = []
    for box in boxes:
        class_id = box_class_id(box)
        class_name = str(names.get(class_id, class_id))
        if class_name != target_label:
            continue
        normalized.append(
            Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=box_confidence(box),
                bbox_xyxy=[float(value) for value in box_xyxy(box)],
            )
        )
    return normalized


def box_class_id(box: Any) -> int | None:
    cls = getattr(box, "cls", None)
    if cls is None:
        return None
    return int(cls.item() if hasattr(cls, "item") else cls[0])


def box_confidence(box: Any) -> float:
    conf = getattr(box, "conf", None)
    if conf is None:
        return 0.0
    return float(conf.item() if hasattr(conf, "item") else conf[0])


def box_xyxy(box: Any) -> list[float]:
    xyxy = getattr(box, "xyxy", None)
    if xyxy is None:
        return [0.0, 0.0, 0.0, 0.0]
    value = xyxy[0] if hasattr(xyxy, "__getitem__") else xyxy
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def empty_detections(session: CaptureSession) -> dict[int, list[Detection]]:
    return {frame_index: [] for frame_index in range(session.frame_count)}


def write_annotation_files(
    session: CaptureSession,
    detections_by_frame: dict[int, list[Detection]],
) -> None:
    for frame_index in range(session.frame_count):
        detections = detections_by_frame.get(frame_index, [])
        labelme_payload = build_labelme_payload(
            image_path=f"../images/{session.frame_name(frame_index)}",
            width=session.recorded_width,
            height=session.recorded_height,
            detections=detections,
        )
        session.label_path(frame_index).write_text(
            json.dumps(labelme_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def run_capture(config: CaptureConfig, status: StatusCallback) -> None:
    rclpy.init(args=None)
    node = GuiServingDatasetCapture(config=config, status=status)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        status("ROS capture stopped")
    finally:
        for thread in node.finalizer_threads:
            thread.join(timeout=30.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        status("Capture thread exited")


class CaptureGui:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.status_queue: queue.Queue[str] = queue.Queue()
        self.capture_thread: threading.Thread | None = None
        self.node: GuiServingDatasetCapture | None = None

        root.title("Serving Dataset Capture")
        root.geometry("760x640")
        root.minsize(700, 580)

        self.vars: dict[str, tk.Variable] = {
            "site_code": tk.StringVar(value=args.site_code),
            "camera_kind": tk.StringVar(value=args.camera_kind),
            "dataset_type": tk.StringVar(value=args.dataset_type),
            "max_events": tk.IntVar(value=args.max_events),
            "output_root": tk.StringVar(value=args.output_root),
            "duration_sec": tk.DoubleVar(value=args.duration_sec),
            "target_fps": tk.DoubleVar(value=args.target_fps),
            "serving_topic": tk.StringVar(value=args.serving_topic),
            "color_topic": tk.StringVar(value=args.color_topic),
            "yolo_model_path": tk.StringVar(value=args.yolo_model_path),
            "conf_threshold": tk.DoubleVar(value=args.conf_threshold),
            "target_label": tk.StringVar(value=args.target_label),
            "skip_yolo": tk.BooleanVar(value=args.skip_yolo),
        }

        self.build()
        self.root.after(100, self.drain_status_queue)

    def build(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)

        naming = ttk.LabelFrame(outer, text="Dataset naming", padding=12)
        naming.grid(row=0, column=0, sticky="ew")
        for column in range(4):
            naming.columnconfigure(column, weight=1)
        self.add_entry(naming, "Site code", "site_code", 0, 0)
        self.add_entry(naming, "Camera kind", "camera_kind", 0, 1)
        self.add_entry(naming, "Type", "dataset_type", 0, 2)
        self.add_entry(naming, "Events to capture", "max_events", 0, 3)

        capture = ttk.LabelFrame(outer, text="Capture", padding=12)
        capture.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        for column in range(2):
            capture.columnconfigure(column, weight=1)
        self.add_entry(capture, "Duration sec", "duration_sec", 0, 0)
        self.add_entry(capture, "FPS", "target_fps", 0, 1)

        topics = ttk.LabelFrame(outer, text="ROS topics", padding=12)
        topics.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        topics.columnconfigure(0, weight=1)
        topics.columnconfigure(1, weight=1)
        self.add_entry(topics, "Serving topic", "serving_topic", 0, 0)
        self.add_entry(topics, "Color topic", "color_topic", 0, 1)

        output = ttk.LabelFrame(outer, text="Output and labels", padding=12)
        output.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        output.columnconfigure(0, weight=1)
        output.columnconfigure(1, weight=1)
        self.add_path_entry(output, "Output root", "output_root", 0)
        self.add_path_entry(output, "YOLO model", "yolo_model_path", 1, directory=False)
        self.add_entry(output, "Target label", "target_label", 2, 0)
        self.add_entry(output, "Conf threshold", "conf_threshold", 2, 1)
        ttk.Checkbutton(
            output,
            text="Skip YOLO and create empty LabelMe JSON",
            variable=self.vars["skip_yolo"],
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        controls = ttk.Frame(outer)
        controls.grid(row=4, column=0, sticky="ew", pady=(14, 8))
        controls.columnconfigure(0, weight=1)
        self.start_button = ttk.Button(controls, text="Start capture", command=self.start_capture)
        self.start_button.grid(row=0, column=1, padx=(0, 8))
        self.stop_button = ttk.Button(
            controls,
            text="Stop",
            command=self.stop_capture,
            state=tk.DISABLED,
        )
        self.stop_button.grid(row=0, column=2)

        log_frame = ttk.LabelFrame(outer, text="Status", padding=8)
        log_frame.grid(row=5, column=0, sticky="nsew")
        outer.rowconfigure(5, weight=1)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

        self.append_log("ROS2 Humble environment loaded from /opt/ros/humble/setup.bash")

    def add_entry(
        self,
        parent: ttk.Frame,
        label: str,
        key: str,
        row: int,
        column: int,
        columnspan: int = 1,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=column, columnspan=columnspan, sticky="ew", padx=4, pady=4)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.vars[key]).grid(row=1, column=0, sticky="ew")

    def add_path_entry(
        self,
        parent: ttk.Frame,
        label: str,
        key: str,
        row: int,
        directory: bool = True,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(frame, textvariable=self.vars[key]).grid(row=1, column=0, sticky="ew")
        ttk.Button(
            frame,
            text="Browse",
            command=lambda: self.browse_path(key, directory),
        ).grid(row=1, column=1, padx=(8, 0))

    def browse_path(self, key: str, directory: bool) -> None:
        if directory:
            selected = filedialog.askdirectory(initialdir=str(Path(self.vars[key].get()).parent))
        else:
            selected = filedialog.askopenfilename(initialdir=str(Path(self.vars[key].get()).parent))
        if selected:
            self.vars[key].set(selected)

    def make_config(self) -> CaptureConfig:
        max_events = int(self.vars["max_events"].get())
        duration_sec = float(self.vars["duration_sec"].get())
        target_fps = float(self.vars["target_fps"].get())
        if max_events < 1:
            raise ValueError("Events to capture must be at least 1")
        if duration_sec <= 0:
            raise ValueError("Duration sec must be positive")
        if target_fps <= 0:
            raise ValueError("FPS must be positive")

        config = CaptureConfig(
            output_root=Path(str(self.vars["output_root"].get())).expanduser(),
            site_code=str(self.vars["site_code"].get()),
            camera_kind=str(self.vars["camera_kind"].get()),
            dataset_type=str(self.vars["dataset_type"].get()),
            max_events=max_events,
            duration_sec=duration_sec,
            target_fps=target_fps,
            serving_topic=str(self.vars["serving_topic"].get()),
            color_topic=str(self.vars["color_topic"].get()),
            yolo_model_path=str(self.vars["yolo_model_path"].get()),
            conf_threshold=float(self.vars["conf_threshold"].get()),
            target_label=str(self.vars["target_label"].get()),
            skip_yolo=bool(self.vars["skip_yolo"].get()),
        )
        return config

    def start_capture(self) -> None:
        if self.capture_thread and self.capture_thread.is_alive():
            return
        try:
            config = self.make_config()
        except Exception as error:
            messagebox.showerror("Invalid capture settings", str(error))
            return

        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.set_form_state(tk.DISABLED)
        self.append_log("Starting ROS capture")

        def status(message: str) -> None:
            self.status_queue.put(message)

        def runner() -> None:
            try:
                run_capture(config, status)
            except Exception as error:
                status(f"Capture failed: {error}")

        self.capture_thread = threading.Thread(target=runner, name="ros-capture", daemon=True)
        self.capture_thread.start()

    def stop_capture(self) -> None:
        self.append_log("Stop requested")
        if rclpy.ok():
            rclpy.shutdown()
        self.stop_button.configure(state=tk.DISABLED)

    def set_form_state(self, state: str) -> None:
        for child in self.root.winfo_children():
            self.set_child_state(child, state)
        if state == tk.DISABLED:
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
        else:
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)

    def set_child_state(self, widget: tk.Widget, state: str) -> None:
        for child in widget.winfo_children():
            if child in (self.start_button, self.stop_button, self.log):
                continue
            try:
                child.configure(state=state)
            except tk.TclError:
                pass
            self.set_child_state(child, state)

    def append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def drain_status_queue(self) -> None:
        while True:
            try:
                message = self.status_queue.get_nowait()
            except queue.Empty:
                break
            self.append_log(message)
        if self.capture_thread and not self.capture_thread.is_alive():
            self.capture_thread = None
            self.append_log("Capture complete; closing GUI")
            self.root.after(800, self.root.destroy)
            return
        self.root.after(100, self.drain_status_queue)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GUI serving-event dataset capture with BBXYZ-style naming."
    )
    parser.add_argument("--no-gui", action="store_true", help="Run directly from CLI.")
    parser.add_argument("--site-code", default="BBXYZ")
    parser.add_argument("--camera-kind", default="topview1")
    parser.add_argument("--dataset-type", default="serving")
    parser.add_argument("--max-events", type=int, default=1)
    parser.add_argument("--serving-topic", default=DEFAULT_SERVING_TOPIC)
    parser.add_argument("--color-topic", default=DEFAULT_COLOR_TOPIC)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--yolo-model-path", default=DEFAULT_YOLO_MODEL_PATH)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--target-label", default="cup")
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Create empty LabelMe JSON files without running YOLO.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> CaptureConfig:
    return CaptureConfig(
        output_root=Path(args.output_root),
        site_code=args.site_code,
        camera_kind=args.camera_kind,
        dataset_type=args.dataset_type,
        max_events=args.max_events,
        duration_sec=args.duration_sec,
        target_fps=args.target_fps,
        serving_topic=args.serving_topic,
        color_topic=args.color_topic,
        yolo_model_path=args.yolo_model_path,
        conf_threshold=args.conf_threshold,
        target_label=args.target_label,
        skip_yolo=args.skip_yolo,
    )


def main() -> None:
    args = parse_args()
    if args.no_gui:
        run_capture(config_from_args(args), print)
        return

    root = tk.Tk()
    CaptureGui(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
