import os 
import json
import traceback
from managers.ConfigManager import ConfigManager
from Modules.CustomLogger import CustomLogger
from objects.ROIObject import ROIObject
class ROIManager:
    def __init__(self,solution_name,camera_id,camera_name,rois):
        try:
            self.solution_name = solution_name
            self.camera_id = camera_id
            self.camera_name = camera_name
            self.config_manager = ConfigManager.get_instance()
            self.name = f"ROIManager"
            self.logger = CustomLogger(self.name).get_logger(
                log_file=f"logs/{self.name}.log",
                log_level=self.config_manager.get("LOG_LEVEL", 10),
                log_to_console=self.config_manager.get("LOG_TO_CONSOLE", True)
            )
            self.logger.info(f"Initializing ROIManager for camera {self.camera_id} with {len(rois) if rois else 0} ROIs.")
            self.rois = rois
            self.updates_in_roi = False
            self.all_roi_objects = []
            self.current_roi_ids, self.current_roi_dict = ({roi['id'] for roi in self.rois}, {roi['id']: roi for roi in self.rois})

        except Exception as exec:
            self.logger.error(f"Error initializing ROIManager for camera {camera_id}: {exec}")
            self.logger.error(traceback.format_exc())
    

    def initialize_rois(self):
        try:
            for roi in self.rois:
                roi_object:ROIObject = ROIObject(
                    roi_id=roi.get("id", "unknown_roi"),
                    name=roi.get("roi_name", "Unnamed ROI"),
                    points=roi.get("points", []),
                    roi_type=roi.get("type", "area_roi")
                )
                self.all_roi_objects.append(roi_object)
            self.logger.info(f"Initialized {len(self.all_roi_objects)} ROIs for camera {self.camera_name=}.")
        
        except Exception as e:
            self.logger.error(f"Error initializing ROIs for camera {self.camera_id}: {e}")
            self.logger.error(traceback.format_exc())

    def convert_all_rois_denorm(self, frame_width: int, frame_height: int):
        try:
            for roi_object in self.all_roi_objects:
                roi_object.denormalize_points(frame_width, frame_height)
            self.logger.info(f"Converted all ROIs to denormalized points based on frame size {frame_width}x{frame_height}.")
        except Exception as e:
            self.logger.error(f"Error converting ROIs to denormalized points for camera {self.camera_id}: {e}")
            self.logger.error(traceback.format_exc())