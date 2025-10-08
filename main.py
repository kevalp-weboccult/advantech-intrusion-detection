"""
Main entry point for the Advantech Intrusion Detection System.

This module initializes the system configuration, sets up logging,
and manages the main application lifecycle.
"""

import faulthandler
import json
import os
import traceback
from typing import Dict, Any, Optional, TextIO

import multiprocessing as mp

from managers.ConfigManager import ConfigManager
from managers.DeviceManager import DeviceManager
from Modules.CustomLogger import CustomLogger

# Suppress OpenCV logging
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = "-8"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MainManager:
    """
    Main application manager for the Advantech Intrusion Detection System.

    Handles system initialization, configuration management, and coordinates
    the overall application lifecycle including device management setup.
    """

    def __init__(self) -> None:
        """
        Initialize the MainManager with configuration and logging setup.

        Sets up multiprocessing, fault handling, device ID management,
        and initializes the DeviceManager instance.
        """
        self.config: ConfigManager = ConfigManager.get_instance()
        self.logger = CustomLogger("MainManager").get_logger(
            log_file="logs/main_manager.log",
            log_level=self.config.defaults.get("LOG_LEVEL", 10),
            log_to_console=self.config.defaults.get("LOG_TO_CONSOLE", True)
        )
        self.logger.info("MainManager initialized with configuration:")

        mp.set_start_method('spawn', force=True)
        # mp.freeze_support()
        self.faulthandler_log_file: str = "faulthandler.log"
        self.faulthandler_log: TextIO = open(
            self.faulthandler_log_file, "w", encoding="utf-8"
        )
        faulthandler.enable(file=self.faulthandler_log)

        # Configuration file to store device ID
        self.config_file: str = "device_data.json"

        # Initialize device ID
        device_id: Optional[str] = self._get_or_create_device_id()
        self.device_id: str = device_id or ""

        self.device_manager: DeviceManager = DeviceManager(self.device_id)

    def _get_or_create_device_id(self) -> Optional[str]:
        """
        Get device ID from config file or prompt user for new one.

        Returns:
            Device ID string if available, None otherwise.
        """
        device_id: Optional[str] = None

        # Check if the configuration file exists
        if not os.path.exists(self.config_file):
            # Prompt user to input device ID if configuration file doesn't exist
            device_id = str(input("Enter device ID: "))
            self.device_id = device_id

            # Save the device ID to the configuration file
            data: Dict[str, str] = {"device_id": device_id}
            with open(self.config_file, "w", encoding="utf-8") as config_file:
                json.dump(data, config_file)
        else:
            # Load the existing device ID from the configuration file
            with open(self.config_file, "r", encoding="utf-8") as config_file:
                data: Dict[str, Any] = json.load(config_file)

            self.device_id = data['device_id']
            self.logger.info(f"Running on device ID: {self.device_id}")
            device_id = data['device_id']

        if device_id:
            self.device_id = device_id

            # Save the updated device ID to the configuration file
            data = {"device_id": device_id}
            with open(self.config_file, "w", encoding="utf-8") as config_file:
                json.dump(data, config_file)
        else:
            self.logger.info(f"Starting with default device ID {self.device_id}")

        return device_id

    def start(self) -> None:
        """
        Start the main application logic.

        This method can be expanded to include additional startup procedures
        as needed.
        """
        self.logger.info("MainManager started.")
        # Additional startup logic can be added here
        try:
            self.device_manager.start()
        except Exception as exec:
            self.logger.error(f"Error starting DeviceManager: {str(exec)}")
            self.logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        main_manager = MainManager()
        main_manager.start()
    except Exception as exec:
        with open("error.log", "a", encoding="utf-8") as error_log:
            error_log.write(f"Exception occurred: {str(exec)}\n")
            error_log.write(traceback.format_exc())
        print(f"An error occurred: {str(exec)}. Check error.log for details.")