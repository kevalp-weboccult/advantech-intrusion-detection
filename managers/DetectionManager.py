import os 
import traceback
from typing import Optional, Dict, Any,List
from managers.ConfigManager import ConfigManager    
from Modules.CustomLogger import CustomLogger
from datetime import datetime, timedelta,timezone
from queue import Queue
from threading import Thread
from shapely.geometry import Point
import numpy as np
import time
import cv2


class DetectionManager:
    def __init__(self,all_roi_objects:List[Any]):
        self.config_manager = ConfigManager.get_instance()
        self.logger = CustomLogger("DetectionManager").get_logger(
            log_file="logs/detection_manager.log",
            log_level=self.config_manager.get("LOG_LEVEL", 10),
            log_to_console=self.config_manager.get("LOG_TO_CONSOLE", True)
        )
        self.alert_interval = timedelta(seconds=self.config_manager.get("ALERT_INTERVAL_SECONDS",300))
        self.logger.info("DetectionManager initialized.")
        self.running:bool = True
        self.reader_queue:Optional[Queue] = None
        self.roi_objects = all_roi_objects
        self.intrusion_normal_buffer = self.config_manager.get("INTRUSION_DETECTED_BUFFER_NORMAL",5)
        self.intrusion_critical_buffer = self.config_manager.get("INTRUSION_DETECTED_BUFFER_CRITICAL",2)
        self.intrusion_normal_time_buffer = timedelta(seconds=self.config_manager.get("INTRUSION_ALERT_INTERVAL_NORMAL",300))
        self.intrusion_critical_time_buffer = timedelta(seconds=self.config_manager.get("INTRUSION_ALERT_INTERVAL_CRITICAL",60))
    def start(self,reader_queue:Queue):
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
                
                frame = data.get("frame")
                h,w,c = frame.shape
                detections = data.get("detections",[])
                inference_frame = data.get("inference_frame",None)
                if frame is None:
                    self.logger.warning("Received None frame in DetectionManager.")
                    continue
                detection_norm_list = []
                # cv2.namedWindow("DetectionManager Inference Frame", cv2.WINDOW_NORMAL)
                # cv2.imshow("DetectionManager Inference Frame", inference_frame)
                # cv2.waitKey(1)
                
                for roi_object in self.roi_objects:
                    roi_object.current_insider_bboxes = []
                    cv2.polylines(frame, [np.array(roi_object.denorm_points).astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
                    self.logger.info(f"{len(detections)=}")
                    for det in detections:
                        bbox = det.get("bbox")
                        self.logger.info(f"Processing detection bbox: {bbox}")
                        # inference_frame = det.get("inference_frame")
                        if bbox is None:
                            continue
                        x1, y1, x2, y2 = bbox
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        point = Point(center_x, center_y)
                        cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)
                        print("aa",roi_object.denorm_poly_points.contains(point))
                        print(roi_object.denorm_poly_points)
                        print("point",point)
                        if roi_object.denorm_poly_points.is_valid and roi_object.denorm_poly_points.contains(point):
                            roi_object.current_insider_bboxes.append(bbox)
                            if self.config_manager.get("DEBUG_MODE", False):
                                self.logger.info(f"Intrusion detected in ROI '{roi_object.name}' for bbox: {bbox}")
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                
                        else:
                            if self.config_manager.get("DEBUG_MODE", False):
                                self.logger.info(f"No intrusion in ROI '{roi_object.name}' for bbox: {bbox}")
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    if len(roi_object.current_insider_bboxes) > 0:
                        # roi_object.last_alert_sent_time = datetime.now(timezone.utc)  
                        current_time = datetime.now(timezone.utc)
                        time_buffer_limit = self.intrusion_normal_time_buffer if roi_object.roi_type == "normal" else self.intrusion_critical_time_buffer
                        if roi_object.last_alert_sent_time is None or (current_time - roi_object.last_alert_sent_time) >= time_buffer_limit     :

                            roi_object.interusion_detection_count += 1
                            buffer_limit = self.intrusion_normal_buffer if roi_object.roi_type == "normal" else self.intrusion_critical_buffer
                            if roi_object.interusion_detection_count >= buffer_limit:
                                roi_object.interusion_detection_count = 0  
                                self.logger.warning(f"Intrusion ALERT in ROI '{roi_object.name=}' with {len(roi_object.current_insider_bboxes)=} objects.")
                                roi_object.last_alert_sent_time = current_time
                    

                        
                    
                #Further processing can be added here
                merged_frame = cv2.hconcat([inference_frame, frame]) if inference_frame is not None else frame
                cv2.namedWindow("DetectionManager Frame", cv2.WINDOW_NORMAL)
                cv2.imshow("DetectionManager Frame", merged_frame)
                cv2.waitKey(1)
                
            except Exception as e:
                self.logger.error(f"Error in DetectionManager loop: {e}")
                self.logger.error(traceback.format_exc())
