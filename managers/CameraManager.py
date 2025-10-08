import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from Modules.CustomLogger import CustomLogger
from managers.ConfigManager import ConfigManager
from managers.APIManger import ApiManager
import numpy as np
from typing import List,TYPE_CHECKING
from classes.Detector import Detector
from objects.CameraObject import CameraObject

class CameraManager:
    def __init__(self,camera_group_id:str,cameras:List[dict],device_id:str,camera_groups,rabbitmq_queue):
        self.camera_group_id = camera_group_id
        self.cameras = cameras
        self.device_id = device_id
        self.name = f"CameraManager-{self.camera_group_id}"
        self.config_manager = ConfigManager.get_instance()
        self.camera_groups = camera_groups
        self.rabbitmq_queue = rabbitmq_queue
        
        self.logger = CustomLogger(self.name).get_logger(
            log_file=f"logs/{self.name}.log",
            log_level=self.config_manager.get("LOG_LEVEL", 10),
            log_to_console=self.config_manager.get("LOG_TO_CONSOLE", True)
        )
        self.logger.info(f"CameraManager initialized for camera group ID: {self.camera_group_id} with {len(self.cameras)} cameras.")
        self.api_manager = ApiManager(self.device_id)

        self.detector = Detector(name=f"Detector-{self.camera_group_id}")

        self.camera_objects: List[CameraObject] = []

        self.__init_cameras(self.cameras)

       

        # Initialize other attributes
        self.running: bool = True
    
    def __init_cameras(self,cameras):
        try:
            for camera in cameras:
                self.camera_objects.append(CameraObject(camera,self))
            self.logger.info(f"Initialized {len(self.camera_objects)} CameraObjects in CameraManager '{self.camera_group_id}'.")
                
        
        except Exception as e:
            self.logger.error(f"Error initializing cameras in CameraManager '{self.camera_group_id}': {e}")
            self.logger.error(traceback.format_exc())
    

    def start(self):
        try:
            for camera_object in self.camera_objects:
                camera_object.start()
            self.logger.info(f"CameraManager '{self.camera_group_id}' started all CameraObjects.")
            while self.running:
                time.sleep(1)
        except Exception as e:
            self.logger.error(f"Error in CameraManager '{self.camera_group_id}': {e}")
            self.logger.error(traceback.format_exc())
    def stop(self):
        try:
            self.running = False
            for camera_object in self.camera_objects:
                camera_object.stop()
            if self.detector:
                self.detector.stop()
            self.logger.info(f"CameraManager '{self.camera_group_id}' stopped all CameraObjects and Detector.")
        except Exception as e:
            self.logger.error(f"Error stopping CameraManager '{self.camera_group_id}': {e}")
            self.logger.debug(traceback.format_exc())
       