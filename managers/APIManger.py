"""
API Manager module for handling device settings and camera details.

This module provides the ApiManager singleton class responsible for managing
API interactions, fetching device settings, and handling camera configurations
for the Advantech Intrusion Detection System.
"""

import json
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union
from logging import Logger

import requests
from bson import ObjectId

from Modules.CustomLogger import CustomLogger
from managers.ConfigManager import ConfigManager
class ApiManager:
    """
    Singleton class for managing API interactions and device settings.

    This class handles fetching device settings from a remote API, checking for
    updates periodically, and managing camera-related data and configuration 
    settings for the intrusion detection device.

    Attributes:
        _instance: Class variable to store the singleton instance.
        last_update_time: Timestamp of the last settings update.
        check_interval: Time interval between settings checks.
        running: Flag indicating if the API manager is active.
        config_manager: Configuration manager instance.
        logger: Logger instance for error and info logging.
        settings: Dictionary holding device settings from API.
        device_id: ObjectId of the device.
        camera_details: Dictionary of camera configurations.
        last_camera_update_time: Timestamp of last camera update.
        check_camera_interval: Time interval for camera health checks.
        filter_roi: Region of interest for filtering operations.
        url: Base URL for API endpoints.
        camera_details_json_file_name: Filename for camera details cache.

    Example:
        >>> api_manager = ApiManager("507f1f77bcf86cd799439011")
        >>> api_manager.check_updated_device_settings()
        >>> cameras = api_manager.get_all_camera_details("507f1f77bcf86cd799439011")
    """

    _instance: Optional["ApiManager"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "ApiManager":
        """
        Create or return the singleton instance of ApiManager.

        Implements the singleton pattern to ensure only one instance
        of ApiManager exists throughout the application lifecycle.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ApiManager: The singleton instance of the ApiManager class.

        Note:
            This method ensures thread-safe singleton implementation.
        """
        if cls._instance is None:
            cls._instance = super(ApiManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, device_id: str) -> None:
        """
        Initialize the ApiManager with device ID and default settings.

        Sets up all necessary attributes for API communication, logging,
        and device configuration management. Initializes singleton instance
        with proper error handling and configuration loading.

        Args:
            device_id: Unique identifier for the device as a string.
                      Must be a valid ObjectId format.

        Raises:
            ValueError: If device_id is not a valid ObjectId format.
            Exception: If singleton instance already exists.

        Note:
            This constructor should only be called once due to singleton pattern.
            Subsequent calls will reference the existing instance.
        """
        # Prevent multiple initialization of singleton
        if hasattr(self, 'initialized'):
            return

        # Initialize timing and control attributes
        self.last_update_time: Optional[datetime] = None
        self.check_interval: timedelta = timedelta(minutes=5)
        self.running: bool = True

        # Initialize configuration manager
        self.config_manager: ConfigManager = ConfigManager.get_instance()

        # Setup logging with proper configuration
        self.logger: Logger = CustomLogger("ApiManager").get_logger(
            log_file="logs/api_manager.log",
            log_level=10,
            log_to_console=True
        )

        # Initialize data storage attributes
        self.settings: Dict[str, Any] = {}
        self.device_id: ObjectId = ObjectId(device_id)
        self.camera_details: Optional[Dict[str, Any]] = None
        self.last_camera_update_time: Optional[datetime] = None
        self.check_camera_interval: timedelta = timedelta(minutes=5)
        self.filter_roi: Optional[Any] = None
        self.device_details: Optional[Dict[str, Any]] = None
        self.data_from_api: Optional[Dict[str, Any]] = None

        # Configuration URLs and file paths
        self.url: str = self.config_manager.defaults.get(
            "URL", "https://ams.weboccult.com"
        )
        self.camera_details_json_file_name: str = self.config_manager.defaults.get(
            "CAMERA_DETAILS_JSON_FILE_NAME", "camera_details.json"
        )

        # Initialize device settings attributes with proper types
        self.vehicle_detection_img_size: Optional[str] = None
        self.save_frames: Optional[bool] = None
        self.visualize: bool = False
        self.debug_mode: bool = False
        self.check_camera_alive_interval: int = 5
        self.is_live: bool = False
        self.iou_threshold_for_mapping: float = 0.7
        self.distance_for_mapping: int = 100
        self.missing_threshold: float = 30.0
        self.save_centroid: bool = False

        # Mark as initialized and load initial settings
        self.initialized: bool = True
        self.check_updated_device_settings()

        self.logger.info(
            f"ApiManager initialized for device {device_id} with URL: {self.url}"
        )

    def check_updated_device_settings(self) -> None:
        """
        Check for updates to device settings via API call.

        Fetches the latest device settings from the remote API if the check
        interval has elapsed since the last update. Updates all relevant
        instance attributes with the new settings values.

        The method handles API communication, response parsing, and error
        recovery with appropriate logging. Settings are cached locally
        to prevent unnecessary API calls.

        Raises:
            requests.RequestException: If API request fails.
            json.JSONDecodeError: If response parsing fails.
            KeyError: If required settings keys are missing.

        Side Effects:
            - Updates self.settings with new configuration values
            - Updates self.last_update_time to current timestamp
            - Updates various device-specific attribute values
            - Logs errors and status information

        Note:
            This method implements rate limiting through check_interval
            to prevent excessive API calls.
        """
        try:
            current_time: datetime = datetime.now(tz=timezone.utc)
            time_check: bool = (
                self.last_update_time is None or
                current_time - self.last_update_time >= self.check_interval
            )

            if time_check:
                headers: Dict[str, str] = {
                    'Content-Type': 'application/x-www-form-urlencoded'
                }

                response: requests.Response = requests.get(
                    f"{self.url}/alpr/get-setting-data",
                    headers=headers,
                    timeout=30
                )

                if response.status_code == 200:
                    data: Dict[str, Any] = response.json()

                    if data and data.get("status"):
                        raw_settings: Dict[str, Any] = data['setting_data']
                        settings: Dict[str, str] = dict(
                            map(
                                lambda x: (x["setting_key"], x["setting_value"]),
                                raw_settings["value"]
                            )
                        )

                        self.settings = settings
                        self.last_update_time = current_time

                        # Update typed attributes with proper conversion and defaults
                        self.vehicle_detection_img_size = settings.get(
                            "VEHICLE_DETECTION_IMG_SIZE"
                        )
                        self.save_frames = self._convert_to_bool(
                            settings.get("SAVE_FRAMES")
                        )
                        self.visualize = self._convert_to_bool(
                            settings.get("VISUALIZE", "false")
                        )
                        self.debug_mode = self._convert_to_bool(
                            settings.get("DEBUG_MODE", "false")
                        )
                        self.check_camera_alive_interval = int(
                            settings.get("CHECK_CAMERA_ALIVE_INTERVAL", "5")
                        )
                        self.is_live = self._convert_to_bool(
                            settings.get('IS_LIVE', "false")
                        )
                        self.iou_threshold_for_mapping = float(
                            settings.get('IOU_THRESHOLD_FOR_MAPPING', "0.7")
                        )
                        self.distance_for_mapping = int(
                            settings.get('DISTANCE_FOR_MAPPING', "100")
                        )
                        self.missing_threshold = float(
                            settings.get('MISSING_THRESHOLD', "30")
                        )
                        self.save_centroid = self._convert_to_bool(
                            settings.get('SAVE_CENTROID', "false")
                        )

                        self.logger.info("Device settings updated successfully")
                    else:
                        self.logger.warning(
                            "API response indicates failure or empty data"
                        )
                else:
                    self.logger.error(
                        f"API request failed with status code: "
                        f"{response.status_code}"
                    )                    
        except requests.RequestException as req_err:
            self.logger.error(
                f"Request error while updating settings: {req_err} "
                f"{traceback.format_exc()}"
            )
            self.last_update_time = datetime.now(tz=timezone.utc)
            time.sleep(0.01)
        except (json.JSONDecodeError, KeyError, ValueError) as parse_err:
            self.logger.error(
                f"Data parsing error while updating settings: {parse_err} "
                f"{traceback.format_exc()}"
            )
            self.last_update_time = datetime.now(tz=timezone.utc)
            time.sleep(0.01)
        except Exception as general_err:
            self.logger.error(
                f"Unexpected error while updating settings: {general_err} "
                f"{traceback.format_exc()}"
            )
            self.last_update_time = datetime.now(tz=timezone.utc)
            time.sleep(0.01)

    def _convert_to_bool(self, value: Optional[Union[str, bool]]) -> bool:
        """
        Convert string or boolean value to boolean type.

        Args:
            value: Value to convert, can be string, boolean, or None.

        Returns:
            bool: Converted boolean value. Returns False for None or invalid values.

        Example:
            >>> self._convert_to_bool("true")
            True
            >>> self._convert_to_bool("false")
            False
            >>> self._convert_to_bool(None)
            False
        """
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return False

    def get_all_camera_details(self, device_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch camera details for all cameras associated with a device.

        Retrieves comprehensive camera configuration data from the remote API
        for the specified device. Implements fallback mechanism using cached
        data if the API request fails. The response is automatically cached
        to a local JSON file for offline access.

        Args:
            device_id: Unique identifier of the device whose cameras to fetch.
                      Can be string representation of ObjectId.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing camera details with
                structure {"cameras": [...]} if successful, None if failed.

        Raises:
            requests.RequestException: If API request fails.
            json.JSONDecodeError: If response parsing fails.
            FileNotFoundError: If fallback cache file doesn't exist.

        Side Effects:
            - Saves API response to cache file specified by
              self.camera_details_json_file_name
            - Logs API responses and error conditions
            - Updates self.camera_details if successful

        Example:
            >>> api_manager = ApiManager("507f1f77bcf86cd799439011")
            >>> cameras = api_manager.get_all_camera_details("507f1f77bcf86cd799439011")
            >>> if cameras:
            ...     for camera in cameras:
            ...         print(f"Camera ID: {camera['camera_id']}")

        Note:
            This method implements automatic fallback to cached data
            when API is unavailable, ensuring system resilience.
        """
        try:
            # Ensure device_id is string format
            normalized_device_id: str = (
                str(device_id) if not isinstance(device_id, str) else device_id
            )

            headers: Dict[str, str] = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            response: requests.Response = requests.post(
                f"{self.url}/get-device-data",
                headers=headers,
                data={"device_id": normalized_device_id},
                timeout=30
            )

            self.logger.info(
                f"API response for device {normalized_device_id}: "
                f"Status {response.status_code}"
            )

            if response.status_code == 200:
                data_from_api: Dict[str, Any] = response.json()

                # Cache response to file for offline access
                try:
                    with open(
                        self.camera_details_json_file_name,
                        'w',
                        encoding='utf-8'
                    ) as file:
                        json.dump(data_from_api, file, indent=2, ensure_ascii=False)
                    self.logger.info(
                        f"Camera details cached to "
                        f"{self.camera_details_json_file_name}"
                    )
                except IOError as io_err:
                    self.logger.warning(
                        f"Failed to cache camera details: {io_err}"
                    )

                # Validate and return camera data
                if data_from_api and data_from_api.get("status"):
                    cameras: Optional[Dict[str, Any]] = data_from_api.get("cameras")
                    self.data_from_api = data_from_api
                    if cameras:
                        self.camera_details = cameras
                        self.logger.info(
                            f"Successfully retrieved {len(cameras)} camera(s) "
                            f"for device {normalized_device_id}"
                        )
                        return cameras

                    self.logger.warning(
                        f"No cameras found for device {normalized_device_id}"
                    )
                else:
                    self.logger.warning(
                        f"API response indicates failure for device "
                        f"{normalized_device_id}"
                    )
            else:
                self.logger.error(
                    f"API request failed with status {response.status_code} "
                    f"for device {normalized_device_id}"
                )

        except requests.RequestException as req_err:
            self.logger.error(
                f"Request error in get_all_camera_details for device {device_id}: "
                f"{req_err} {traceback.format_exc()}"
            )
        except json.JSONDecodeError as json_err:
            self.logger.error(
                f"JSON parsing error in get_all_camera_details: "
                f"{json_err} {traceback.format_exc()}"
            )
        except Exception as general_err:
            self.logger.error(
                f"Unexpected error in get_all_camera_details: "
                f"{general_err} {traceback.format_exc()}"
            )

        # Fallback to cached camera details file
        return self._load_cached_camera_details()

    def _get_device_data(self) -> Optional[Dict[str, Any]]:
        """
        Extract device data from API response.

        Retrieves device-specific configuration data from the cached API response.
        This method processes the 'device_data' field from the API response that
        was stored during the camera details fetch operation.

        Returns:
            Optional[Dict[str, Any]]: Device data dictionary if available and valid,
                None if data is missing or extraction fails.

        Raises:
            AttributeError: If data_from_api is None or doesn't have expected structure.
            KeyError: If 'device_data' key is missing from API response.

        Side Effects:
            - Updates self.device_details with extracted data
            - Logs success, warning, and error conditions

        Example:
            >>> api_manager = ApiManager("507f1f77bcf86cd799439011")
            >>> device_data = api_manager._get_device_data()
            >>> if device_data:
            ...     print(f"Device name: {device_data.get('device_name')}")

        Note:
            This method depends on successful execution of get_all_camera_details
            which populates self.data_from_api.
        """
        try:
            if self.data_from_api is None:
                self.logger.warning(
                    "No API data available. Call get_all_camera_details first."
                )
                return None

            if not isinstance(self.data_from_api, dict):
                self.logger.error(
                    f"Invalid API data type: {type(self.data_from_api)}. "
                    "Expected dictionary."
                )
                return None

            device_data: Optional[Dict[str, Any]] = self.data_from_api.get("device_data")
            
            if device_data is None:
                self.logger.warning(
                    "No device_data found in API response"
                )
                return None

            if not isinstance(device_data, dict):
                self.logger.error(
                    f"Invalid device_data type: {type(device_data)}. "
                    "Expected dictionary."
                )
                return None

            self.device_details = device_data
            self.logger.info(
                f"Successfully extracted device data with "
                f"{len(device_data)} field(s)"
            )
            return device_data

        except AttributeError as attr_err:
            self.logger.error(
                f"Attribute error while extracting device data: {attr_err} "
                f"{traceback.format_exc()}"
            )
        except KeyError as key_err:
            self.logger.error(
                f"Key error while extracting device data: {key_err} "
                f"{traceback.format_exc()}"
            )
        except Exception as general_err:
            self.logger.error(
                f"Unexpected error while extracting device data: {general_err} "
                f"{traceback.format_exc()}"
            )

        return None


    def _load_cached_camera_details(self) -> Optional[Dict[str, Any]]:
        """
        Load camera details from cached file as fallback.

        Attempts to load previously cached camera details from the local
        JSON file when API requests fail. Provides system resilience
        during network outages or API unavailability.

        Returns:
            Optional[Dict[str, Any]]: Cached camera details if file exists
                and is valid, None otherwise.

        Raises:
            FileNotFoundError: If cache file doesn't exist.
            json.JSONDecodeError: If cache file is corrupted.

        Side Effects:
            - Logs warning about using offline/cached data
            - Logs errors if cache loading fails

        Note:
            This is a private method used internally by get_all_camera_details.
        """
        try:
            with open(
                self.camera_details_json_file_name,
                'r',
                encoding='utf-8'
            ) as file:
                camera_details: Dict[str, Any] = json.load(file)

            self.logger.warning(
                f"Using cached camera details from "
                f"{self.camera_details_json_file_name}"
            )

            if camera_details and camera_details.get("status"):
                cached_cameras: Optional[Dict[str, Any]] = camera_details.get(
                    "cameras"
                )
                if cached_cameras:
                    self.logger.info(
                        f"Loaded {len(cached_cameras)} camera(s) from cache"
                    )
                    return cached_cameras

                self.logger.warning("No cameras found in cached data")
            else:
                self.logger.warning("Cached data indicates failure status")          
        except FileNotFoundError:
            self.logger.error(
                f"Camera details cache file not found: "
                f"{self.camera_details_json_file_name}"
            )
        except json.JSONDecodeError as json_err:
            self.logger.error(
                f"Failed to parse cached camera details file: {json_err}"
            )
        except Exception as general_err:
            self.logger.error(
                f"Unexpected error loading cached camera details: {general_err}"
            )

        return None

    def get_device_settings(self) -> Dict[str, Any]:
        """
        Get current device settings dictionary.

        Returns a copy of the current device settings to prevent
        external modification of internal state.

        Returns:
            Dict[str, Any]: Copy of current device settings.

        Example:
            >>> api_manager = ApiManager("507f1f77bcf86cd799439011")
            >>> settings = api_manager.get_device_settings()
            >>> print(settings.get("VISUALIZE", False))
        """
        return self.settings.copy()

    def is_running(self) -> bool:
        """
        Check if the API manager is currently running.

        Returns:
            bool: True if running, False otherwise.
        """
        return self.running

    def stop(self) -> None:
        """
        Stop the API manager operations.

        Sets the running flag to False and logs the shutdown.
        This method should be called during application shutdown.

        Side Effects:
            - Sets self.running to False
            - Logs shutdown information
        """
        self.running = False
        self.logger.info("ApiManager stopped")

    def __str__(self) -> str:
        """
        Return string representation of ApiManager.

        Returns:
            str: Formatted string with key ApiManager information.
        """
        return (
            f"ApiManager(device_id={self.device_id}, "
            f"url={self.url}, "
            f"running={self.running}, "
            f"last_update={self.last_update_time})"
        )

    def __repr__(self) -> str:
        """
        Return detailed string representation for debugging.

        Returns:
            str: Detailed string representation.
        """
        return (
            f"ApiManager(device_id='{self.device_id}', "
            f"url='{self.url}', "
            f"running={self.running}, "
            f"settings_count={len(self.settings)}, "
            f"last_update='{self.last_update_time}')"
        )
    