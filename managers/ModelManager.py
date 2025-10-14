from __future__ import annotations
import base64
import os
import json
import platform
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import GPUtil
import getmac
import psutil
import requests
from tqdm import tqdm
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from utils.device_utils import check_the_system_and_get_uuid
from utils.encryption_utils import encrypt_file, decrypt_file
from utils.file_utils import delete_files
import logging
from Modules.CustomLogger import CustomLogger
from managers.ConfigManager import ConfigManager

class DeviceInfo:
    """Handles fetching device details like CPU, GPU, RAM, and Device Address."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initializes the DeviceInfo module and retrieves system details.

        Args:
            logger (logging.Logger): Logger instance for logging system information.
        """
        self.logger:logging.Logger = logger
        self.logger.info("Initializing DeviceInfo module...")
        self.device_data: Dict[str, Any] = self._get_device_details()

    def _get_device_details(self) -> Dict[str, Any]:
        """Fetches CPU, GPU, RAM, and Device address details of the system.

        Returns:
            Dict[str, Any]: Dictionary containing system details (CPU, GPUs, RAM,Device Address).
        """
        self.logger.info("Retrieving system details...")
        data: Dict[str, Any] = {}

        # Get Device Address (Unique Device ID)
        data["MAC_Address"]: str = check_the_system_and_get_uuid() # type: ignore

        # Get CPU Details
        data["CPU"]: str = platform.processor() # type: ignore
        self.logger.info(f"CPU details retrieved successfully. Details: {data['CPU']}")

        # Get GPU Details
        try:
            gpus = GPUtil.getGPUs()
            data["GPUs"]: Dict[str, int] = {} #{gpu.name: gpu.memoryTotal for gpu in gpus} # type: ignore
            self.logger.info(f"GPU details retrieved successfully. Details: {data['GPUs']}")
        except Exception as e:
            self.logger.error(f"Error retrieving GPU details: {e}")
            data["GPUs"]: Dict[str, int] = {} # type: ignore

            
        # Get RAM Information
        data["RAM"]: int = psutil.virtual_memory().total # type: ignore
        self.logger.info(f"RAM details retrieved successfully. Details: {data['RAM']}")

        self.logger.info("System details retrieved successfully.")
        return data


class EncryptionManager:
    """Handles key generation, encryption, and decryption of model files."""

    def __init__(self, device_address: str) -> None:
        """Initializes the encryption manager with a derived encryption key.

        Args:
            mac_address (str): The MAC address of the device.
        """
        self.key: bytes = self._derive_key_from_mac(device_address)
        self.config_manager = ConfigManager.get_instance()
        self.encrpytion_file_path = self.config_manager.get("ENCRYPTED_FILE_PATH","_internal/cache/temp.txt")
        # self.logger: logging.Logger = CustomLogger("EncryptionManager").get_logger(
        #     log_file="logs/encryption_manager.log",
        #     log_to_console=True,
        #     log_level=logging.INFO,
        # )

    def _derive_key_from_mac(self, device_address: str) -> bytes:
        """Generates an encryption/decryption key from the MAC address.

        Args:
            mac_address (str): The MAC address of the device.

        Returns:
            bytes: A 32-byte encryption key.
        """
        device_address = device_address.lower().replace(":", "")

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 32-byte key for strong encryption
            salt=b'mac_salt',  # A fixed salt for consistent key generation
            iterations=100000,
            backend=default_backend()
        )

        key: bytes = kdf.derive(device_address.encode())
        return base64.urlsafe_b64encode(key)

    def encrypt(self, file_data: str) -> bool:
        """Encrypts data and saves it to a file.

        Args:
            file_data (str): The data to be encrypted.

        Returns:
            bool: True if encryption is successful, False otherwise.
        """
        return encrypt_file(self.key, None, file_data, self.encrpytion_file_path)

    def decrypt(self) -> Optional[bytes]:
        """Decrypts data from a file.

        Returns:
            Optional[bytes]: Decrypted data if successful, None otherwise.
        """
        return decrypt_file(self.key, self.encrpytion_file_path, None)

class ModelDownloader:
    """Handles downloading and validating AI models."""

    def __init__(self, logger: Any, encryption_manager: EncryptionManager, device_data: Dict[str, Any]) -> None:
        self.logger = logger
        self.config_manager = ConfigManager.get_instance()    
        self.model_folder = self.config_manager.get("ENCRYPTED_MODELS_PATH","MODELS")
        self.detection_model_name = self.config_manager.get("DETECTION_MODEL_NAME","best4.onnx")
        self.encryption_manager = encryption_manager
        self.device_data = device_data
        self.detection_model_path = self.config_manager.get("DETECTION_MODEL_PATH","best4.onnx")
        self.detection_model_path = os.path.join(self.model_folder,self.detection_model_path)
        self.required_paths: list[str] = [
            self.detection_model_path
        ]
        self.ENCRYPTED_MODELS_PATH = self.config_manager.get("ENCRYPTED_MODELS_PATH","_internal/cache/models")
        self.logger.info((f"Required paths: {self.required_paths}"))
        self.token = self.config_manager.get("TOKEN","")
        self.token_api_url = self.config_manager.get("TOKEN_API_URL","https://ams.weboccult.com/token/verify")
        self.get_device_data_api_url = self.config_manager.get("GET_DETVICE_DATA_API_URL","https://app.gotilo.ai")

    def _models_exist(self) -> bool:
        """Checks if all required model files are available."""
        return all(os.path.exists(path) for path in self.required_paths)

    def _fetch_salt_keys(self) -> Dict[str, str]:
        """Fetches model salts and downloads models if missing."""
        salt_keys_list: Dict[str, str] = {}

        if not self._models_exist():
            print("Some required files are missing... Re-downloading models.")
            delete_files(self.required_paths)
            # print("TOKEN",TOKEN)
            data: Dict[str, str] = {"token": self.token, "device_details": json.dumps(self.device_data)}
            response = requests.post(url=self.token_api_url, data=data)
            response_data: Dict[str, Any] = response.json()

            if response_data["code"] != 202:
                print(f"ERROR: {response_data}")
                self.logger.critical(f"Error fetching data: {response_data}")
                exit(0)

            salt_keys_list = self._download_models(response_data)

            file_data: str = json.dumps(response_data)
            if not self.encryption_manager.encrypt(file_data):
                delete_files(self.required_paths)

        else:
            file_data: bytes | None = self.encryption_manager.decrypt()
            if file_data is None:
                delete_files(self.required_paths)

            json_data: Dict[str, Any] = json.loads(file_data.decode('utf-8'))
            expiration_date: datetime = datetime.strptime(
            json_data["expiration_date"], "%Y-%m-%dT%H:%M:%S.%fZ"   
            ).replace(tzinfo=timezone.utc)
            print(type(expiration_date))
            if  datetime.now(tz=timezone.utc) > expiration_date:
                delete_files(self.required_paths)
                print("TOKEN EXPIRED... Contact Admin!")
                self.logger.critical("TOKEN EXPIRED... Contact Admin!")
                exit(0)

            for model in json_data["models_urls"]:
                model_name_splitted: str = model["model_name"].split("__")[0]
                salt_keys_list[model_name_splitted] = model["salt"]

        return salt_keys_list

    def _download_models(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Downloads encrypted AI models."""
        self.logger.info("Downloading AI models...")
        salt_keys: Dict[str, str] = {}

        for model in tqdm(data["models_urls"], desc="Downloading Models"):
            model_name: str = model["model_name"]
            model_salt: str = model["salt"]
            model_name_splitted: str = model_name.split("__")[0]
            salt_keys[model_name_splitted] = model_salt

            model_filename: str = f"{model_name_splitted}.bin"
            model_url: str = model["model_url"][0]

            if not os.path.exists(self.ENCRYPTED_MODELS_PATH):
                os.makedirs(self.ENCRYPTED_MODELS_PATH)

            model_path: str = os.path.join(self.ENCRYPTED_MODELS_PATH, model_filename)

            response = requests.get(model_url, stream=True)
            response.raise_for_status()

            with open(model_path, "wb") as model_file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        model_file.write(chunk)

        return salt_keys
class ModelManager:
    """Main class managing the AI model lifecycle."""

    def __init__(self) -> None:
        self.config_manager = ConfigManager.get_instance()
        # self.logger = 
        self.logger: logging.Logger = CustomLogger("ModelManager").get_logger(
            log_file="logs/model_manager.log",
            log_to_console=self.config_manager.get("log_to_console", True),
            log_level=self.config_manager.get("log_level", logging.INFO),
        )
        self.logger.info("Initializing ModelManager...")
        try:
            self.device_info: DeviceInfo = DeviceInfo(self.logger)
            self.encryption_manager: EncryptionManager = EncryptionManager(
                self.device_info.device_data["MAC_Address"]
            )
            self.model_downloader: ModelDownloader = ModelDownloader(
                self.logger, self.encryption_manager, self.device_info.device_data
            )

            self.salt_keys_list: Dict[str, str] = self.model_downloader._fetch_salt_keys()
            self.logger.info(f"Salt keys list: {self.salt_keys_list}")
            self.logger.info("ModelManager initialized successfully.")
        except Exception as e:
            self.logger.critical(f"Error initializing ModelManager: {e}")
            self.logger.critical(traceback.format_exc())
            raise
    def start(self) -> None:
        """Starts the model management process."""
        self.logger.info("Starting ModelManager...")

    def stop(self) -> None:
        """Stops ModelManager and handles cleanup."""
        self.logger.info("Stopping ModelManager...")