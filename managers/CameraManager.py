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
from utils.encryption_utils import decrypt_models
from typing import Dict,Any,Optional,List 
from dotenv import load_dotenv
load_dotenv(override=True)
if TYPE_CHECKING:
    from multiprocessing import Queue
class CameraManager:
    def __init__(self,camera_group_id:str,cameras:List[dict],device_id:str,camera_groups,rabbitmq_queue,model_manager=None):
        self.camera_group_id = camera_group_id
        self.cameras = cameras
        self.device_id = device_id
        self.name = f"CameraManager-{self.camera_group_id}"
        self.config_manager = ConfigManager.get_instance()
        self.camera_groups = camera_groups
        self.rabbitmq_queue = rabbitmq_queue
        self.model_manager = model_manager
        
        self.logger = CustomLogger(self.name).get_logger(
            log_file=f"logs/{self.name}.log",
            log_level=self.config_manager.get("LOG_LEVEL", 10),
            log_to_console=self.config_manager.get("LOG_TO_CONSOLE", True)
        )
        self.detection_model_path = os.environ.get("DETECTION_MODEL_PATH",self.config_manager.get("DETECTION_MODEL_PATH","WOT03-det-640-20251013.bin"))
        self.encrpyted_models_path = self.config_manager.get("ENCRYPTED_MODELS_PATH","_internal/cache/models")
        self.detection_model_full_path = os.path.join(self.encrpyted_models_path,self.detection_model_path)
        self.detection_model_type = self.config_manager.get("DETECTION_MODEL_TYPE","onnx")
        self.all_model_full_paths = {
            "onnx": self.detection_model_full_path,
        }
        self.all_model_save_path= []
        for model_type, path in self.all_model_full_paths.items():
            model_name = os.path.basename(path).split('.')[0]
            model_decrypted_data = decrypt_models(key=self.model_manager.salt_keys_list[model_name],input_file=path,output_file=None)
            model_save_path = os.path.join(self.encrpyted_models_path, f"{model_name}_decrypted.{model_type}")
            with open(model_save_path, 'wb') as f:
                f.write(model_decrypted_data)
            self.all_model_save_path.append(model_save_path)
        





        self.logger.info(f"CameraManager initialized for camera group ID: {self.camera_group_id} with {len(self.cameras)} cameras.")
        self.api_manager = ApiManager(self.device_id)

        self.detector = Detector(name=f"Detector", model_path=self.all_model_save_path[0])
        os.remove(self.all_model_save_path[0])

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
       