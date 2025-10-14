import os
import traceback

import cv2
from classes.Steamer import Streamer
from Modules.CustomLogger import CustomLogger
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, TYPE_CHECKING
from managers.ConfigManager import ConfigManager
from managers.ROIManger import ROIManager
from managers.DetectionManager import DetectionManager
from threading import Thread
if TYPE_CHECKING:
    from managers.CameraManager import CameraManager
class CameraObject:
    def __init__(self, camera_details: dict, camera_manager:"CameraManager"):
        self.camera_details = camera_details
        self.camera_manager = camera_manager
        self.device_id = self.camera_manager.device_id
        self.camera_id = camera_details.get("_id", "unknown_camera")
        self.name = f"CameraObject"
        self.config_manager = ConfigManager.get_instance()
        self.detector = self.camera_manager.detector

        self.logger = CustomLogger(self.name).get_logger(
            log_file=f"logs/{self.name}.log",
            log_level=self.config_manager.get("LOG_LEVEL", 10),
            log_to_console=self.config_manager.get("LOG_TO_CONSOLE", True)
        )
        self.logger.info(f"CameraObject {self.camera_id} initialized.")
        self.rois = None
        self.start_time = None
        self.rabbitmq_queue = self.camera_manager.rabbitmq_queue


        try:
            self.rois = camera_details.get("rois", [])
        except Exception as exec:
            self.logger.error(f"There are no rois for the camera")
        self.roi_manager = ROIManager(solution_name="IntrusionDetection",camera_id=self.camera_id,camera_name=camera_details.get("camera_name","Unnamed Camera"),rois=self.rois)
        self.roi_manager.initialize_rois()
        self.all_roi_objects=self.roi_manager.all_roi_objects
        self.streamer: Optional[Streamer] = Streamer(name=f"Streamer", camera_data=camera_details,all_roi_objects = self.all_roi_objects,rabbitmq_queue=self.rabbitmq_queue)
        width = self.streamer.resize_width if self.streamer.is_resize else self.streamer.frame_width
        height = self.streamer.resize_height if self.streamer.is_resize else self.streamer.frame_height
        
        self.roi_manager.convert_all_rois_denorm(frame_width=width,frame_height=height)

        self.detection_manager = DetectionManager(all_roi_objects=self.all_roi_objects,camera_details=self.camera_details,rabbitmq_queue=self.rabbitmq_queue)
        self.streamer_thread = Thread(target=self.streamer.start, daemon=True)
        self.detector_thread = Thread(target=self.detector.start,args=(self.streamer.writer_queue,) ,daemon=True)
        self.detectiion_manager_thread = Thread(target=self.detection_manager.start,args=(self.detector.writer_queue,) ,daemon=True)
        self.running: bool = True
        self.intrusion_alert_interval_normal = timedelta(seconds=self.config_manager.get("INTRUSION_ALERT_INTERVAL_NORMAL",300))
        self.intrusion_alert_interval_critical = timedelta(seconds=self.config_manager.get("INTRUSION_ALERT_INTERVAL_CRITICAL",60))
        
        
       

        self.site_id = camera_details["site_id"]
        self.device_id = camera_details["device_id"]


        

    
    def display_thread(self)-> None:
        try:
            while self.running:
                if not self.tracker.writer_queue.empty():
                    result = self.detector.writer_queue.get()
                    inference_frame = result['inference_frame']
                    detections = result['detections']
                    cv2.imshow(f"Camera {self.camera_id}", inference_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            cv2.destroyAllWindows()
        except Exception as exec:
            self.logger.error(f"Error in display thread of CameraObject {self.camera_id}: {exec}")
            self.logger.error(traceback.format_exc())


    def start(self) -> None:
        try:
            self.streamer_thread.start()
            self.detector_thread.start()
            self.detectiion_manager_thread.start()
            # self.tracker_thread.start()
            # Thread(target=self.display_thread,daemon=True).start()
            self.logger.info(f"CameraObject {self.camera_id} started streamer and detector threads.")
        except Exception as exec:
            self.logger.error(f"Error starting CameraObject {self.camera_id}: {exec}")
            self.logger.error(traceback.format_exc())
        
    def stop(self) -> None:
        try:
            self.running = False
            if self.streamer:
                self.streamer.stop()
            if self.detector:
                self.detector.stop()
            self.streamer_thread.join()
            self.detector_thread.join()
            self.logger.info(f"CameraObject {self.camera_id} stopped successfully.")
        except Exception as e:
            self.logger.error(f"Error stopping CameraObject {self.camera_id}: {e}")
            self.logger.error(traceback.format_exc())


       
