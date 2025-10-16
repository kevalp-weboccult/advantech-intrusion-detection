import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import multiprocessing as mp
from uuid import uuid4

from Modules.CustomLogger import CustomLogger
from managers.ConfigManager import ConfigManager
from managers.APIManger import ApiManager
from typing import Any
from utils.device_utils import check_if_the_object_pickleable
from managers.CameraManager import CameraManager
from managers.RabbitmqManager import RabbitmqManager
from threading import Thread
from managers.ModelManager import ModelManager
class DeviceManager:
    def __init__(self,token:str,name:str="device_manager"):
        self.token = token
        self.name = name
        self.config_manager = ConfigManager.get_instance()
        
        self.logger = CustomLogger(self.name).get_logger(
            log_file=f"logs/{self.name}.log",
            log_level=self.config_manager.get("LOG_LEVEL", 10),
            log_to_console=self.config_manager.get("LOG_TO_CONSOLE", True)
        )
        # self.logger.info(f"DeviceManager initialized for device ID: {self.device_id}")
        self.api_manager = ApiManager(self.token)
        self.all_camera_details: Optional[Dict[str,Any]] = self.api_manager.get_all_camera_details()
        self.device_id = self.api_manager.device_id if self.api_manager.device_id else "unknown_device"
        self.device_data:Optional[Dict[str,Any]] = self.api_manager._get_device_data()


        self.running: bool = True
        self.last_update_camera_details_time: datetime = datetime.now(tz=timezone.utc)
        self.check_interval: timedelta = timedelta(minutes=5)
        self.device_status_check_interval: timedelta = timedelta(minutes=3)
        self.last_device_status_check_time: Optional[datetime] = None
        self.model_manager: Optional[ModelManager] = ModelManager()

        self.RABBITMQ_WRITER_QUEUE_SIZE:int = self.config_manager.get("RABBITMQ_WRITER_QUEUE_SIZE",500)
        self.total_processes:List[mp.Process] = []

        # Shared state for inter-process communication
        manager: mp.Manager = mp.Manager()
        self.camera_groups: Any = manager.dict()
        self.shared_device_dict = manager.dict()
        self.rabbitmq_queue = manager.Queue(maxsize=self.RABBITMQ_WRITER_QUEUE_SIZE)


        # Local dictionary for process objects
        self.processes: Dict[str, mp.Process] = {}
        self.CAMERAS_PER_PROCESS = self.config_manager.get("CAMERAS_PER_PROCESS", 3)

        self.rmq_process: Optional[mp.Process] = mp.Process(
            target=start_rabbitmq_manager_process,
            args=(self.rabbitmq_queue,self.camera_groups,self.shared_device_dict,self.logger),
        )
        
        self.device_status_update_thread:Optional[Thread] = Thread(target=self.send_device_heartbeat_message,daemon=True)

        # Initialize device update process
        

        # Initialize camera managers
        self.initialize_and_start_camera_manager()

        self.last_device_details_update_time: Optional[datetime] = None
        self.device_update_interval: timedelta = timedelta(minutes=5)
        self.rabbitmq_queue_name:str = self.config_manager.get("RABBITMQ_QUEUE", "intrusion_event_logs")
        
    

    def send_device_heartbeat_message(self) -> None:
        try:
            while self.running:
                try:
                    current_time:Optional[datetime] = datetime.now(timezone.utc)
                    if self.last_device_details_update_time is None or current_time - self.last_device_details_update_time >= self.device_update_interval:
                        device_id = self.device_id
                        self.device_name = self.device_data.get("device_name", "Unnamed Device") if self.device_data else "Unnamed Device"
                        site_id = self.all_camera_details[0].get("site_id","") if self.all_camera_details and len(self.all_camera_details)>0 else ""
                        message = {
                            "site_id": site_id,
                            "device_id": device_id,
                            "device_name": self.device_name,
                            "current_time": datetime.now(timezone.utc).isoformat(),
                        }
                        device_heartbeat_message = {
                            "queue_name": self.rabbitmq_queue_name,
                            "message_type": "heartbeat_device",
                            "message": message
                        }
                        self.rabbitmq_queue.put(device_heartbeat_message)   
                        self.last_device_details_update_time = current_time
                    time.sleep(10)  # Sleep to prevent excessive CPU usage
                
                except Exception as e:
                    self.logger.error(f"Error in device heartbeat loop: {str(e)}")
                    self.logger.error(traceback.format_exc())
        
        except Exception as exec:
            self.logger.error(f"Error sending device heartbeat message: {str(exec)}")
            self.logger.error(traceback.format_exc())
                
    def initialize_and_start_camera_manager(self) -> None:
        try:
            for i in range(0, len(self.all_camera_details), self.CAMERAS_PER_PROCESS):
                camera_group_id: str = str(uuid4())  # Unique group ID
                cameras: List[Dict] = self.all_camera_details[i:i + self.CAMERAS_PER_PROCESS]  # Camera group
                cameras_data: List[Dict] = [camera.copy() for camera in cameras]  # Deep copy for safety

                # Ensure all objects are pickleable
                assert check_if_the_object_pickleable(self.device_id)
                assert check_if_the_object_pickleable(cameras)

                # Store camera data in shared state
                self.camera_groups[camera_group_id] = {
                    "Cameras": cameras_data,
                    "TotalCameras": len(cameras_data),
                    "CameraIDs": [camera["_id"] for camera in cameras_data],
                }

                # Start process
                process: mp.Process = mp.Process(
                    target=start_camera_manager_process,
                    args=(camera_group_id, cameras_data, self.device_id, self.camera_groups, self.rabbitmq_queue,self.model_manager),
                )
                self.processes[camera_group_id] = process
        
        except Exception as e:
            self.logger.error(f"Error initializing camera managers: {str(e)}/n{traceback.format_exc()}")
    
    
    def start(self) -> None:
        try:
            self.device_status_update_thread.start()
            for group_id, process in self.processes.items():
                self.logger.info(f"Starting process for group {group_id}...")
                process.start()
                self.total_processes.append(process)
                self.logger.info(f"Camera group {group_id} started.")
            self.rmq_process.start()  
            self.total_processes.append(self.rmq_process)
            
            # Start device update process
            self.logger.info("Starting device update process...")
            
            try:
                for p in self.total_processes:
                    p.join()
                while self.running:
                    self.logger.info("DeviceManager heartbeat...")
                    time.sleep(10)

            
            except Exception as exec:
                self.logger.error(f"Error in joining processes: {str(exec)}")
                self.logger.error(traceback.format_exc())
        except Exception as e:
            self.logger.error(f"Error starting camera manager processes: {e}")
            self.logger.error(traceback.format_exc())
    
    def stop(self) -> None:
        self.running = False
        for group_id, process in self.processes.items():
            if process.is_alive():
                process.terminate()
                self.logger.info(f"Camera group {group_id} stopped.")
        
        # Stop device update process
        if self.device_update_process and self.device_update_process.is_alive():
            self.device_update_process.terminate()
            self.logger.info("Device update process stopped.")
            
        # Stop RabbitMQ process
        if self.rmq_process and self.rmq_process.is_alive():
            self.rmq_process.terminate()
            self.logger.info("RabbitMQ process stopped.")
            
        self.logger.info("DeviceManager stopped.")


def start_camera_manager_process(camera_group_id: str, cameras: List[dict], device_id: str, camera_groups: Any,rabbitmq_queue,model_manager:ModelManager) -> None:
    try:
        camera_manager = CameraManager(camera_group_id, cameras, device_id, camera_groups,rabbitmq_queue,model_manager)
        camera_manager.start()
        while camera_manager.running:
            time.sleep(1)
    except Exception as e:
        logger = CustomLogger(f"CameraManager-{camera_group_id}").get_logger(
            log_file=f"logs/CameraManager-{camera_group_id}.log",
            log_level=10,
            log_to_console=True
        )
        logger.error(f"Error in CameraManager process {camera_group_id}: {str(e)}/n{traceback.format_exc()}")
        raise

def start_rabbitmq_manager_process(rabbitmq_queue,shared_camera_dict,shared_device_dict,logger):
        try:
            logger.info(f"Starting RabbitMQManager for group", flush=True)
            rabbitmq_manager = RabbitmqManager(rabbitmq_queue,shared_camera_dict,shared_device_dict)
            rabbitmq_manager.start()
            logger.info(f"RabbitMQManager started", flush=True)
            logger.info(f"RabbitMQManager started")
            
        except Exception as e:
            logger.error(f"Error in RabbitMQ Manager process : {e} {traceback.format_exc()}")

