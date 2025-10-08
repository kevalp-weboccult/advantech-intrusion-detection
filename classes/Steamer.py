import time
import traceback
from typing import Optional, Dict, Any, Union
from logging import Logger
from queue import Queue

import cv2
import numpy as np
from numpy.typing import NDArray

from managers.ConfigManager import ConfigManager
from Modules.CustomLogger import CustomLogger
from utils.image_utils import preprocess_frame,create_blackout_image
from datetime import datetime, timedelta, timezone
# Type aliases for clarity
FrameType = NDArray[np.uint8]
FrameDataType = Dict[str, Any]


class Streamer:
    """
    A video stream reader with queue-based frame publishing.

    
   
    """

    def __init__(self, name: str = "streamer", camera_data=None,all_roi_objects=None,rabbitmq_queue=None) -> None:
        """
        it initializes the Streamer with a name and camera URL.
        Handles:
        - Robust stream initialization and restart logic.
        - FPS throttling and custom frame dropping.
        - Metadata tracking (frame count, dimensions, fps).
        - Queue-based producer for downstream consumers.
        """
        self.name: str = name
        self.config_manager: ConfigManager = ConfigManager.get_instance()

        # Logger setup
        self.logger: Logger = CustomLogger(self.name).get_logger(
            log_file=f"logs/{name}.log",
            log_level=self.config_manager.get("LOG_LEVEL"),
            log_to_console=self.config_manager.get("LOG_TO_CONSOLE"),
        )

        self.logger.info(f"Initializing Streamer '{self.name}' with settings: {self.config_manager.defaults}")
        self.stream_url = camera_data['stream_url']
        self.site_id = camera_data['site_id']
        self.camera_id = camera_data['_id']
        self.camera_name = camera_data['camera_name']
        self.device_id = camera_data['device_id']
        self.resize = camera_data['is_resize']
        self.rotation = camera_data['rotate']
        self.rabbitmq_queue = rabbitmq_queue

        self.resolution_x = camera_data["resolution_x"]
        self.resolution_y = camera_data["resolution_y"]
        self.is_resize = camera_data["is_resize"]
        self.resize_width = camera_data["resize_width"] if self.is_resize else self.resolution_x
        self.resize_height = camera_data["resize_height"] if self.is_resize else self.resolution_y
        self.all_roi_objects = all_roi_objects

        try:
            self.stream_url = int(self.stream_url)
        except Exception as e:
            self.stream_url = self.stream_url
        
        self.stream = cv2.VideoCapture(self.stream_url)

        
        self.fps: int = int(self.stream.get(cv2.CAP_PROP_FPS))
        self.width: int = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.last_updated_time:datetime =  None
        self.camera_heartbeat_timeout = timedelta(minutes=self.config_manager.get("CAMERA_HEARTBEAT_TIMEOUT_MINS",2))




        

        

        self.ret_false_count: int = 0
        self.frame_count: int = 0
        self.running: bool = True

        self.writer_queue: Queue[FrameDataType] = Queue(maxsize=1)

        self.max_ret_false_count: int = self.config_manager.get("MAX_RET_FALSE_COUNT", 10)
        self.is_live: bool = self.config_manager.get("IS_LIVE", True)
        self.custom_fps: bool = self.config_manager.get("CUSTOM_FPS", False)
        self.target_fps: int = self.config_manager.get("FPS", 5)
        self.rabbitmq_queue_name = self.config_manager.get("RABBITMQ_QUEUE", "intrusion_event_logs")

        self.logger.info(
            f"Streamer '{self.name}' initialized "
            f"(Resolution={self.width}x{self.height}, FPS={self.fps}, Live={self.is_live} , resize=({self.resize_width}x{self.resize_height}) , rotation={self.rotation})"
        )

  
    def _read_frame(self) -> Optional[FrameType]:
        """Attempt to read a frame from the stream."""
        ret, frame = self.stream.read()
        frame = cv2.imread("test_2.jpg")
        # frame = cv2.imread("/media/wot-keval/New Volume/ai_projects/python_projects/digital_twin/7th_floor/frame_1851.jpg")
        frame = preprocess_frame(frame, self.rotation, self.is_resize, self.resize_width, self.resize_height)
        if not ret:
            self.ret_false_count += 1
            self.logger.warning(
                f"Failed to read frame ({self.ret_false_count}/{self.max_ret_false_count})"
            )
            if self.ret_false_count >= self.max_ret_false_count:
                self._restart_stream()
            return None
        
        self.ret_false_count = 0
        current_time = datetime.now(timezone.utc)
        if self.last_updated_time is None or current_time - self.last_updated_time > self.camera_heartbeat_timeout:
            message = {
                "site_id": self.site_id,
                "device_id": self.device_id,
                "camera_id": self.camera_id,
                "timestamp": current_time.isoformat(),
                "fps":5
            }
            camera_heartbeat_message = {
                "queue_name" :self.rabbitmq_queue_name,
                "message_type": "heartbeat",
                "message": message
            }

            try:
                if self.rabbitmq_queue and not self.rabbitmq_queue.full():
                    self.rabbitmq_queue.put(camera_heartbeat_message)
                    self.logger.info(f"Camera heartbeat sent for camera {self.camera_id}")
                    self.last_updated_time = current_time
                else:
                    self.logger.warning("RabbitMQ queue is full or not initialized. Heartbeat not sent.")
            except Exception as e:
                self.logger.error(f"Failed to send camera heartbeat: {e} | {traceback.format_exc()}")
        return frame

    def _wrap_metadata(self, frame: FrameType,blackout_frame) -> FrameDataType:
        """Attach metadata to a frame."""
        return {
            "frame": frame,
            "blackout_frame": blackout_frame,
            "frame_count": self.frame_count,
            "timestamp": time.time(),
            "dimensions": (self.width, self.height),
        }

    def _restart_stream(self) -> None:
        """Restart the video stream when too many failures occur."""
        try:
            self.logger.info("Restarting video stream...")
            self.stream.release()
            time.sleep(2)
            # self.stream = cv2.VideoCapture(self.config_manager.get("CAMERA_URL", 0))
            # self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            # self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            # self.fps = float(self.stream.get(cv2.CAP_PROP_FPS) or 0.0)
            try:
                self.stream_url = int(self.stream_url)
            except Exception as e:
                self.stream_url = self.stream_url
            self.stream = cv2.VideoCapture(self.stream_url)
            self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self.fps = float(self.stream.get(cv2.CAP_PROP_FPS) or 0.0)

                              
            
            self.ret_false_count = 0
            self.frame_count = 0
            self.logger.info(
                f"Stream restarted successfully "
                f"(Resolution={self.width}x{self.height}, FPS={self.fps})"
            )
        except Exception as e:
            self.logger.error(f"Error restarting stream: {e} | {traceback.format_exc()}")

  
    def start(self) -> None:
        """Main loop for reading and publishing frames."""
        self.logger.info("Streamer started.")
        while self.running:
            try:
                frame = self._read_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                blackout_frame, original_frame = create_blackout_image(frame, self.all_roi_objects)
                frame_data = self._wrap_metadata(frame,blackout_frame)

                # FPS throttling
                if self.custom_fps and (self.frame_count % self.target_fps != 0):
                    continue

                # Queue handling
                if not self.is_live:
                    while self.writer_queue.full() and self.running:
                        time.sleep(0.05)

                if not self.writer_queue.full():
                    self.writer_queue.put(frame_data)
                    self.logger.debug(f"Frame {frame_data['frame_count']} queued.")

            except Exception as e:
                self.logger.error(f"Error in stream loop: {e} | {traceback.format_exc()}")
                time.sleep(1)

    def stop(self) -> None:
        """Stop the streamer gracefully."""
        self.running = False
        self.release()
        time.sleep(0.2)
        self.logger.info("Streamer stopped.")

    def release(self) -> None:
        """Release the video stream resource."""
        if self.stream and self.stream.isOpened():
            self.stream.release()
            self.logger.info("Stream released.")
        else:
            self.logger.warning("Stream already released or not opened.")

    def info(self) -> None:
        """Log the current streamer configuration and state."""
        self.logger.info("=" * 40)
        self.logger.info(f"Streamer Name: {self.name}")
        self.logger.info(f"Camera URL: {self.config_manager.get('CAMERA_URL', 0)}")
        self.logger.info(f"Resolution: {self.width}x{self.height}")
        self.logger.info(f"FPS (reported): {self.fps}")
        self.logger.info(f"Frame Count: {self.frame_count}")
        self.logger.info(f"Is Live: {self.is_live}")
        self.logger.info(f"Custom FPS: {self.custom_fps} -> {self.target_fps}")
        self.logger.info(f"Queue Size: {self.writer_queue.qsize()}")
        self.logger.info(f"Running: {self.running}")
        self.logger.info("=" * 40)

    def __str__(self) -> str:
        return (
            f"Streamer(name={self.name}, url={self.config_manager.get('CAMERA_URL', 0)}, "
            f"resolution={self.width}x{self.height}, fps={self.fps}, "
            f"frame_count={self.frame_count}, live={self.is_live}, "
            f"max_ret_false_count={self.max_ret_false_count}, running={self.running})"
        )