import os
import traceback
from typing import Optional, Dict, Any, List
from managers.ConfigManager import ConfigManager    
from Modules.CustomLogger import CustomLogger
from datetime import datetime, timedelta, timezone
from queue import Queue
from shapely.geometry import Point
import numpy as np
import time
import cv2
from utils.image_utils import convert_to_base64
from utils.bbox_utils import normalize_bbox

class DetectionManager:
    def __init__(self, all_roi_objects: List[Any], camera_details: Optional[Dict[str, Any]] = None, rabbitmq_queue: Optional[Queue] = None):
        self.config_manager = ConfigManager.get_instance()
        self.logger = CustomLogger("DetectionManager").get_logger(
            log_file="logs/detection_manager.log",
            log_level=self.config_manager.get("LOG_LEVEL", 10),
            log_to_console=self.config_manager.get("LOG_TO_CONSOLE", True)
        )
        self.alert_interval = timedelta(seconds=self.config_manager.get("ALERT_INTERVAL_SECONDS", 300))
        self.logger.info("DetectionManager initialized.")
        self.running: bool = True
        self.reader_queue: Optional[Queue] = None
        self.roi_objects = all_roi_objects
        self.intrusion_normal_buffer = self.config_manager.get("INTRUSION_DETECTED_BUFFER_NORMAL", 5)
        self.intrusion_critical_buffer = self.config_manager.get("INTRUSION_DETECTED_BUFFER_CRITICAL", 2)
        self.intrusion_normal_time_buffer = timedelta(seconds=self.config_manager.get("INTRUSION_ALERT_INTERVAL_NORMAL", 300))
        self.intrusion_critical_time_buffer = timedelta(seconds=self.config_manager.get("INTRUSION_ALERT_INTERVAL_CRITICAL", 60))
        self.camera_details = camera_details
        self.camera_id = camera_details.get("_id", "unknown_camera") if camera_details else "unknown_camera"
        self.camera_name = camera_details.get("camera_name", "unknown_camera") if camera_details else "unknown_camera"
        self.site_id = camera_details.get("site_id", "unknown_site") if camera_details else "unknown_site"
        self.rabbitmq_queue_name = self.config_manager.get("RABBITMQ_QUEUE", "intrusion_event_log_queue")
        self.rabbitmq_queue = rabbitmq_queue

    def start(self, reader_queue: Queue):
        self.reader_queue = reader_queue
        
        while self.running:
            try:
                if self.reader_queue.empty():
                    time.sleep(0.1)
                    continue
                data = self.reader_queue.get()
                if data is None:
                    time.sleep(0.1)
                    continue

                s_time = time.time()
                frame = data.get("frame")
                org_image = frame.copy() if frame is not None else None
                h, w, c = frame.shape
                detections = data.get("detections", [])
                inference_frame = data.get("inference_frame", None)

                if frame is None:
                    self.logger.warning("Received None frame in DetectionManager.")
                    continue

                # Process each ROI
                for roi_object in self.roi_objects:
                    roi_object.current_insider_bboxes = []
                    roi_type = "normal_roi" if roi_object.roi_type == "normal_roi" else "critical_roi"
                    # Draw ROI polygons
                    color_roi = (255, 0, 0) if roi_type == "normal_roi" else (0, 0, 255)  # Blue for normal, Red for critical
                    cv2.polylines(frame, [np.array(roi_object.denorm_points).astype(np.int32)], isClosed=True, color=color_roi, thickness=2)

                    for det in detections:
                        bbox = det.get("bbox")
                        if bbox is None:
                            continue
                        x1, y1, x2, y2 = bbox
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        point = Point(center_x, center_y)
                        print(f"Processing bbox: {bbox} with center point: ({center_x}, {center_y})")
                        # Check if detection is inside ROI
                        if roi_object.denorm_poly_points.is_valid and roi_object.denorm_poly_points.contains(point):
                            roi_object.current_insider_bboxes.append(bbox)
                            print(f"Detection inside ROI '{roi_type}': {bbox}")
                            bbox_color = (255, 0, 0) if roi_type == "normal_roi" else (0, 0, 255)  # Blue if normal, Red if critical
                        else:
                            print(f"Detection outside ROI '{roi_type}': {bbox}")
                            bbox_color = (0, 255, 0)  # Green if outside ROI

                        # Draw bbox and center point
                        print(f"Drawing bbox: {bbox} with color: {bbox_color}")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                        cv2.circle(frame, (int(center_x), int(center_y)), 5, bbox_color, -1)

                    self.logger.info(f"ROI '{roi_object.name}' detected {len(roi_object.current_insider_bboxes)} insider bboxes.")

                    # Handle alerts based on buffer/time
                    if len(roi_object.current_insider_bboxes) > 0:
                        self.logger.info(f"Intrusion detected in ROI '{roi_object.name}'.")
                        current_time = datetime.now(timezone.utc)
                        time_buffer_limit = self.intrusion_normal_time_buffer if roi_object.roi_type == "normal" else self.intrusion_critical_time_buffer
                        if roi_object.last_alert_sent_time is None or (current_time - roi_object.last_alert_sent_time) >= time_buffer_limit:
                            roi_object.interusion_detection_count += 1
                            buffer_limit = self.intrusion_normal_buffer if roi_object.roi_type == "normal" else self.intrusion_critical_buffer
                            alert_type = "normal" if roi_object.roi_type == "normal" else "critical"
                            if roi_object.interusion_detection_count >= buffer_limit:
                                roi_object.interusion_detection_count = 0
                                roi_object.last_alert_sent_time = current_time
                                base64_image = convert_to_base64(org_image)
                                message  = {
                                    "camera_name": self.camera_name,
                                    "camera_id": self.camera_id,
                                    "image_url": base64_image,
                                    "site_id": self.site_id,
                                    "is_resolved": False,
                                    "alert_type": alert_type,
                                    "location": roi_object.roi_id,
                                    "event_time": current_time.isoformat(),
                                }
                                event_data = {
                                    "queue_name": self.rabbitmq_queue_name,
                                    "message_type": "intrusion_event_log",
                                    "message": message
                                }
                                self.rabbitmq_queue.put(event_data)
                                
                                self.logger.info(f"Sending intrusion alert: {event_data['message_type']}")

                    else:
                        roi_object.interusion_detection_count = 0

                e_time = time.time()
                merged_frame = cv2.hconcat([inference_frame, frame]) if inference_frame is not None else frame
                cv2.namedWindow("DetectionManager Frame", cv2.WINDOW_NORMAL)
                cv2.imshow("DetectionManager Frame", frame)
                cv2.waitKey(0)

            except Exception as e:
                self.logger.error(f"Error in DetectionManager loop: {e}")
                self.logger.error(traceback.format_exc())
