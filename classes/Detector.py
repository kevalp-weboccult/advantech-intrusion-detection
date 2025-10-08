"""Object detection module for Person Identifier AI application.

This module provides a robust object detection system using YOLO (You Only Look Once)
architecture through the Ultralytics library. It supports real-time object detection,
configurable detection parameters, and queue-based processing pipeline.

Features:
- YOLO-based object detection with multiple model support
- Configurable confidence and IoU thresholds
- Queue-based multi-threaded processing
- Real-time detection visualization
- Automatic model warmup for optimal performance
- Live and buffered processing modes
- Comprehensive logging and error handling

Author: WOT
Version: 1.0.0
License: MIT
"""

# Standard library imports
from __future__ import annotations

import os
import time
import traceback
from logging import Logger
from queue import Queue
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Third-party imports
import numpy as np
from numpy.typing import NDArray
import cv2

# Local application imports
from managers.ConfigManager import ConfigManager
from Modules.CustomLogger import CustomLogger
from Modules.ONNXDetector import ONNXDetector

# Type checking imports
if TYPE_CHECKING:
    from ultralytics import YOLO

# Type aliases for better code readability
FrameType = NDArray[np.uint8]
BboxType = List[int]
DetectionType = Dict[str, Any]
FrameDataType = Dict[str, Any]

class Detector:
    """Advanced object detection system using YOLO architecture.
    
    This class implements a comprehensive object detection system that processes
    video frames in real-time using YOLO models. It provides configurable detection
    parameters, queue-based processing, and visualization capabilities.
    
    Key Features:
    - YOLO model integration with Ultralytics library
    - Configurable confidence and IoU thresholds
    - Real-time detection visualization
    - Queue-based multi-threaded processing
    - Live and buffered processing modes
    - Automatic model warmup for optimal performance
    - Comprehensive logging and error handling
    - Support for custom model paths and configurations
    
    Attributes:
        name (str): Unique identifier for this detector instance
        settings_manager (ConfigManager): Configuration manager singleton
        logger (Logger): Logger instance for this detector
        iou (float): Intersection over Union threshold for NMS
        model_path (str): Path to the YOLO model file
        yolo_model (Optional[YOLO]): YOLO model instance
        confidence (float): Minimum confidence threshold for detections
        reader_queue (Optional[Queue]): Input queue for frames to process
        writer_queue (Queue): Output queue for processed frames
        running (bool): Flag indicating if the detector is active
        inference_frame (Optional[FrameType]): Current inference frame for visualization
        MODEL_IMGSZ (List[int]): Model input image size [width, height]
        is_live (bool): Whether to operate in live mode (drop frames if queue full)
        
    Example:
        >>> detector = Detector("main_detector")
        >>> detector.start(input_queue)
        >>> # Process frames...
        >>> detector.stop()
    """
    
    def __init__(
            self,
            name: str = "detector",
            yolo_model: Optional["YOLO"] = None
    ) -> None:
        """Initialize the Detector with configuration and model loading.
        
        Sets up the YOLO model, configuration management, logging, and all
        necessary parameters for robust object detection operation.
        
        Args:
            name (str, optional): Unique identifier for this detector instance.
                Used for logging and identification. Defaults to "detector".
            yolo_model (Optional[YOLO], optional): Pre-initialized YOLO model.
                If None, a new model will be loaded from the configured path.
                Defaults to None.
                
        Raises:
            Exception: If model loading fails or configuration is invalid.
        """
        try:
            # Core configuration
            self.name: str = name
            self.settings_manager: ConfigManager = ConfigManager.get_instance()
            
            # Initialize logging
            self.logger: Logger = CustomLogger(self.name)
            self.logger = self.logger.get_logger(
                log_file=f"logs/{name}.log",
                log_level=self.settings_manager.get("LOG_LEVEL"),
                log_to_console=self.settings_manager.get("LOG_TO_CONSOLE"),
            )
            
            self.logger.info(f"Initializing Detector '{self.name}'...")
            self.use_yolo: bool = self.settings_manager.get("USE_YOLO", False)
            self.detector_queue_size: int = self.settings_manager.get("DETECTOR_QUEUE_SIZE", 5)
            
            # Initialize detector and yolo_model attributes
            self.detector: Optional[ONNXDetector] = None
            self.yolo_model: Optional["YOLO"] = None
            
            # Detection parameters
            self.iou: float = self.settings_manager.get("IOU_THRESHOLD", 0.5)
            self.confidence: float = self.settings_manager.get("CONF_THRESHOLD", 0.5)
            self.model_imgsz: List[int] = self.settings_manager.get("MODEL_IMGSZ", [640, 640])
            self.is_live: bool = self.settings_manager.get("IS_LIVE", True)
            
            # Model setup
            self.model_path: str = self.settings_manager.get(
                "DETECTION_MODEL_PATH", 
                "models/yolov8n.pt"
            )
            if self.use_yolo:
                self._initialize_yolo_model(yolo_model)
            else:
                self.detector = ONNXDetector()
                self.yolo_model = None
            # Queue initialization
            self.reader_queue: Optional["Queue[FrameDataType]"] = None
            self.writer_queue: "Queue[FrameDataType]" = Queue(maxsize=self.detector_queue_size)
            self.running: bool = True
            
            # Visualization
            self.inference_frame: Optional[FrameType] = None
            
            # Validate configuration
            self._validate_configuration()
            
            # Perform warmup if model is available
            if self.use_yolo and self.yolo_model is not None:
                self.logger.info(f"Detector '{self.name}' model classes: {self.yolo_model.names}")
                self.warmup(warmup_steps=5)
            
            self.logger.info(
                f"Detector '{self.name}' initialized successfully - "
                f"Confidence: {self.confidence}, IoU: {self.iou}, "
                f"Image size: {self.model_imgsz}, Live mode: {self.is_live}"
            )
            
        except Exception as exc:
            error_msg = f"Error initializing Detector '{name}': {exc}"
            print(f"{error_msg}\nTraceback: {traceback.format_exc()}")
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
    
    def _initialize_yolo_model(self, yolo_model: Optional["YOLO"]) -> None:
        """Initialize YOLO model with fallback handling.
        
        Args:
            yolo_model: Pre-initialized YOLO model or None.
            
        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If model initialization fails.
        """
        from ultralytics import YOLO
        if yolo_model is not None:
            self.yolo_model = yolo_model
            self.logger.info("Using provided YOLO model instance")
        else:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"YOLO model file not found: {self.model_path}")
            
            try:
                self.yolo_model = YOLO(self.model_path)
                self.logger.info(f"YOLO model loaded successfully from: {self.model_path}")
            except Exception as exc:
                error_msg = f"Failed to load YOLO model from {self.model_path}: {exc}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg) from exc
    
    def _validate_configuration(self) -> None:
        """Validate detector configuration parameters.
        
        Raises:
            ValueError: If configuration parameters are invalid.
        """
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {self.confidence}")
        
        if not (0.0 <= self.iou <= 1.0):
            raise ValueError(f"IoU threshold must be between 0.0 and 1.0, got {self.iou}")
        
        if len(self.model_imgsz) != 2 or any(size <= 0 for size in self.model_imgsz):
            raise ValueError(
                f"Model image size must be [width, height] with positive values, "
                f"got {self.model_imgsz}"
            )
        
        self.logger.debug("Configuration validation completed successfully")
    
    def warmup(self, warmup_steps: int = 5) -> None:
        """Perform model warmup to optimize inference performance.
        
        Runs several dummy inferences to initialize model optimizations
        and GPU memory allocation. This reduces latency for the first real
        inference calls.
        
        Args:
            warmup_steps: Number of warmup inference steps to perform.
                
        Note:
            Warmup is particularly important for GPU-based models to allocate
            memory and compile kernels.
        """
        if not self.use_yolo or self.yolo_model is None:
            self.logger.warning("Cannot perform warmup: YOLO model not available")
            return
        
        self.logger.info(f"Starting warmup of YOLO model with {warmup_steps} steps...")
        
        try:
            # Create dummy input image with random data
            dummy_image: FrameType = np.random.randint(
                0, 255,
                (self.model_imgsz[1], self.model_imgsz[0], 3),
                dtype=np.uint8
            )
            
            # Perform warmup inferences
            for i in range(warmup_steps):
                self.yolo_model.predict(
                    dummy_image,
                    iou=self.iou,
                    conf=self.confidence,
                    verbose=False,
                    imgsz=self.model_imgsz
                )
                self.logger.debug(f"Warmup step {i + 1}/{warmup_steps} completed")
            
            self.logger.info("YOLO model warmup completed successfully")
            
        except Exception as exc:
            error_msg = f"Error during model warmup: {exc}"
            self.logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
            # Don't raise exception as warmup failure shouldn't stop initialization
    
    def draw_detections_on_frame(
            self,
            frame: FrameType,
            detections: Optional[List[DetectionType]]
    ) -> FrameType:
        """Draw detection bounding boxes and labels on frame.
        
        Visualizes object detections by drawing bounding boxes, class IDs,
        and confidence scores on the input frame.
        
        Args:
            frame (FrameType): Input frame as numpy array to draw on.
            detections (Optional[List[DetectionType]]): List of detection
                dictionaries containing bbox, confidence, and class_id.
                If None or empty, returns the original frame unchanged.
                
        Returns:
            FrameType: Frame with detection visualizations drawn on it.
            
        Note:
            This method modifies the input frame directly and also returns it.
            Bounding boxes are drawn in red (BGR: 0,0,255) with thickness 2.
        """
        if not detections:
            return frame
        
        try:
            for detection in detections:
                bbox: BboxType = detection["bbox"]
                class_id: int = detection["class_id"]
                conf: float = detection["confidence"]

                # Prepare text label
                text: str = f"ID:{class_id} | {conf:.2f}"
                
                # Get class name if available
                if (self.use_yolo and self.yolo_model is not None and
                        hasattr(self.yolo_model, 'names')):
                    class_name: str = self.yolo_model.names.get(
                        class_id, f"Class_{class_id}"
                    )
                    text = f"{class_name} | {conf:.2f}"
                
                # Draw text label above bounding box
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1] - text_size[1] - 10),
                    (bbox[0] + text_size[0], bbox[1]),
                    (0, 0, 255),
                    -1
                )
                cv2.putText(
                    frame,
                    text,
                    (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 0, 255),
                    2
                )
            
            self.logger.debug(f"Drew {len(detections)} detections on frame")
            
        except Exception as exc:
            self.logger.error(f"Error drawing detections: {exc}")
        
        return frame
    
    def get_detection_data(self, frame: FrameType) -> Optional[List[DetectionType]]:
        """Extract object detection data from input frame.
        
        Performs object detection on the input frame using the YOLO model
        and processes the results into a standardized format.
        
        Args:
            frame (FrameType): Input frame as numpy array for object detection.
            
        Returns:
            Optional[List[DetectionType]]: List of detection dictionaries
                containing bbox coordinates, confidence scores, and class IDs.
                Returns None if detection fails or no model is available.
                
        Note:
            Each detection dictionary contains:
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - confidence: detection confidence score (0.0 to 1.0)
            - class_id: integer class identifier
        """
        # if not self.use_yolo:
        #     self.logger.error("Cannot perform detection: YOLO model not available")
        #     return None
        
        try:
            if self.use_yolo and self.yolo_model is not None:
                detections: List[DetectionType] = []

                # Run YOLO inference
                results = self.yolo_model.predict(
                    frame,
                    iou=self.iou,
                    conf=self.confidence,
                    verbose=False,  # Changed to False to reduce log spam
                    imgsz=self.model_imgsz
                )[0]

                # Store inference frame for visualization
                self.inference_frame = frame.copy()

                # Extract detection data
                if results.boxes is not None:
                    bbox_datas = results.boxes.data
                    self.logger.debug(f"Raw detections found: {len(bbox_datas)}")

                    for bbox_data in bbox_datas:
                        bbox_data = bbox_data.cpu().numpy()

                        # Skip detections below confidence threshold
                        if bbox_data[4] < self.confidence:
                            continue

                        # Extract coordinates and metadata
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2, conf, cls = bbox_data
                        bbox_x1 = int(bbox_x1)
                        bbox_y1 = int(bbox_y1)
                        bbox_x2 = int(bbox_x2)
                        bbox_y2 = int(bbox_y2)

                        # Validate bounding box coordinates
                        if bbox_x1 >= bbox_x2 or bbox_y1 >= bbox_y2:
                            self.logger.warning(
                                f"Invalid bbox coordinates: "
                                f"[{bbox_x1}, {bbox_y1}, {bbox_x2}, {bbox_y2}]"
                            )
                            continue

                        # Create detection dictionary
                        detection: DetectionType = {
                            "bbox": [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                            "confidence": float(conf),
                            "class_id": int(cls),
                        }
                        detections.append(detection)
            else:
                detections: List[DetectionType] = []
                self.inference_frame = frame.copy()
                start_time_ns = time.time_ns()
                boxes, scores, class_ids = self.detector.detect(frame)
                end_time_ns = time.time_ns()
                # Detection timing for performance monitoring
                detection_time_ms = (end_time_ns - start_time_ns) / 1e6
                self.logger.debug(f"ONNX detection time: {detection_time_ms:.2f} ms")

                for box, score, class_id in zip(boxes, scores, class_ids):
                    self.inference_frame = self.detector.draw_detections(
                        self.inference_frame, box, score, class_id
                    )
                    self.logger.info(f"Box: {box}, Score: {score}, Class ID: {class_id}")
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = box
                    bbox_x1 = int(bbox_x1)
                    bbox_y1 = int(bbox_y1)
                    bbox_x2 = int(bbox_x2)
                    bbox_y2 = int(bbox_y2)
                    # self.logger.info(f"Box: {box}, Score: {score}, Class ID: {class_id}")
                    detection: DetectionType = {
                        "bbox": [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                        "confidence": float(score),
                        "class_id": int(class_id),
                    }
                    # print(detection)
                    self.logger.info(f"detection: {detection=}")
                    detections.append(detection)
            return detections

        except Exception as exc:
            error_msg = f"Error getting detection data: {exc}"
            self.logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
            return None

    def start(self, reader_queue: "Queue[FrameDataType]") -> None:
        """Start the object detection processing loop.

        Main processing loop that reads frames from the input queue, performs
        object detection, draws visualizations, and puts results in the output queue.

        Args:
            reader_queue: Input queue containing frame
                data dictionaries with at least a 'frame' key.

        Note:
            This method runs in an infinite loop until self.running is set to False.
            It processes frames continuously and handles exceptions gracefully.
        """
        try:
            self.reader_queue = reader_queue
            self.logger.info(
                f"Detector '{self.name}' started with reader queue. "
                f"Live mode: {self.is_live}"
            )

            while self.running:
                try:
                    # Check for available frames
                    if self.reader_queue.empty():
                        time.sleep(0.1)
                        continue

                    # Get frame data from input queue
                    frame_data: FrameDataType = self.reader_queue.get()
                    if frame_data is None:
                        time.sleep(0.1)
                        continue

                    # Extract frame from data
                    frame: Optional[FrameType] = frame_data.get('frame')
                    if frame is None:
                        self.logger.warning(
                            "Received frame data without 'frame' key, skipping..."
                        )
                        continue
                    
                    # Perform object detection with timing
                    start_time: float = time.time()
                    detections: Optional[List[DetectionType]] = self.get_detection_data(frame)
                    end_time: float = time.time()
                    detection_time: float = end_time - start_time

                    # Add detection results to frame data
                    frame_data['detections'] = detections

                    # Create visualization frame
                    if self.inference_frame is not None:
                        # Add frame count to visualization
                        frame_count: int = frame_data.get('frame_count', 0)
                        cv2.putText(
                            self.inference_frame,
                            f"Frame: {frame_count}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                        frame_data['inference_frame'] = self.inference_frame
                    
                    # Handle queue management based on live/buffered mode
                    if not self.is_live:
                        # Buffered mode: wait for queue space
                        while self.writer_queue.full() and self.running:
                            time.sleep(0.1)

                    # Add frame to output queue if space available
                    if not self.writer_queue.full():
                        self.writer_queue.put(frame_data)
                        self.logger.debug(
                            f"Detection completed: {len(detections) if detections else 0} "
                            f"objects in {detection_time:.3f}s "
                            f"(Queue size: {self.writer_queue.qsize()})"
                        )
                    elif self.is_live:
                        # Live mode: drop frame if queue is full
                        self.logger.debug("Writer queue full, dropping frame in live mode")
                
                except Exception as frame_error:
                    self.logger.error(
                        f"Error processing frame in detector loop: {frame_error}\n"
                        f"Traceback: {traceback.format_exc()}"
                    )
                    # Brief pause before continuing to prevent error spam
                    time.sleep(0.1)

            self.logger.info(f"Detector '{self.name}' processing loop terminated")

        except Exception as exc:
            error_msg = f"Fatal error in Detector start method: {exc}"
            self.logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from exc
    

    def stop(self) -> None:
        """Stop the detector gracefully and clean up resources.

        Sets the running flag to False to terminate the processing loop
        and performs any necessary cleanup operations.

        Note:
            This method is thread-safe and can be called from any thread
            to request shutdown of the detection processing.
        """
        try:
            self.logger.info(f"Stopping Detector '{self.name}'...")
            self.running = False

            # Allow time for the processing loop to exit gracefully
            time.sleep(0.2)

            # Note: YOLO models from ultralytics don't require explicit cleanup
            # but we could add it here if needed in the future

            self.logger.info(f"Detector '{self.name}' stopped successfully")

        except Exception as exc:
            error_msg = f"Error stopping Detector '{self.name}': {exc}"
            self.logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")

    def get_writer_queue(self) -> "Queue[FrameDataType]":
        """Get reference to the output queue for processed frames.

        Returns:
            The output queue containing processed
                frame data with detection results added.

        Note:
            This queue contains frame data dictionaries with added 'detections'
            and 'inference_frame' keys after processing.
        """
        return self.writer_queue

    def get_processed_frame(self, identifier: str = "") -> Optional[FrameDataType]:
        """Get the most recent processed frame from the output queue.

        Retrieves the latest processed frame data without removing it from
        the queue. Useful for monitoring or display purposes.

        Args:
            identifier: Identifier for logging purposes.

        Returns:
            The most recent frame data dictionary
                if available, None if the queue is empty.

        Note:
            This method does not remove the frame from the queue, so multiple
            calls will return the same frame until new data is processed.
        """
        if self.writer_queue.empty():
            self.logger.debug(
                f"No processed frames available for {identifier or self.name}"
            )
            return None

        try:
            # Peek at the first item without removing it
            return self.writer_queue.queue[0]
        except (IndexError, AttributeError):
            self.logger.warning("Error accessing writer queue")
            return None

    def update_confidence_threshold(self, new_confidence: float) -> None:
        """Update the confidence threshold for detections.

        Args:
            new_confidence: New confidence threshold between 0.0 and 1.0.

        Raises:
            ValueError: If the confidence value is not between 0.0 and 1.0.
        """
        if not 0.0 <= new_confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {new_confidence}"
            )

        old_confidence = self.confidence
        self.confidence = new_confidence
        self.logger.info(
            f"Updated confidence threshold from {old_confidence} to {new_confidence}"
        )

    def update_iou_threshold(self, new_iou: float) -> None:
        """Update the IoU threshold for Non-Maximum Suppression.

        Args:
            new_iou: New IoU threshold between 0.0 and 1.0.

        Raises:
            ValueError: If the IoU value is not between 0.0 and 1.0.
        """
        if not 0.0 <= new_iou <= 1.0:
            raise ValueError(f"IoU must be between 0.0 and 1.0, got {new_iou}")

        old_iou = self.iou
        self.iou = new_iou
        self.logger.info(f"Updated IoU threshold from {old_iou} to {new_iou}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded YOLO model.

        Returns:
            Dictionary containing model information including
                class names, model path, and current configuration.
        """
        info = {
            "name": self.name,
            "model_path": self.model_path,
            "confidence_threshold": self.confidence,
            "iou_threshold": self.iou,
            "model_image_size": self.model_imgsz,
            "is_live_mode": self.is_live,
            "running": self.running,
        }

        if self.use_yolo and self.yolo_model is not None:
            info.update({
                "model_classes": self.yolo_model.names,
                "num_classes": (
                    len(self.yolo_model.names) if self.yolo_model.names else 0
                ),
            })
        else:
            info.update({
                "model_classes": None,
                "num_classes": 0,
            })

        return info

    def __str__(self) -> str:
        """Return a string representation of the Detector instance.

        Returns:
            str: Formatted string containing key detector properties
                and current state information.
        """
        return (
            f"Detector("
            f"name='{self.name}', "
            f"model_path='{self.model_path}', "
            f"confidence={self.confidence}, "
            f"iou={self.iou}, "
            f"image_size={self.MODEL_IMGSZ}, "
            f"live_mode={self.is_live}, "
            f"running={self.running}, "
            f"model_loaded={self.use_yolo and self.yolo_model is not None}"
            f")"
        )

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging.

        Returns:
            Detailed string representation suitable for debugging.
        """
        return self.__str__()