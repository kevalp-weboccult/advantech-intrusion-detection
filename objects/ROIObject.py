import os 
import traceback

import json
from datetime import datetime, timedelta, timezone
from Modules.CustomLogger import CustomLogger
from shapely.geometry import Point, Polygon


class ROIObject:
    def __init__(self, roi_id: str, name: str, points: list,roi_type):
        self.roi_id = roi_id
        self.name = name
        self.norm_points = points  # List of (x, y) tuples
        self.roi_type = roi_type  
        self.created_at = datetime.now(tz=timezone.utc)
        self.updated_at = datetime.now(tz=timezone.utc)
        self.update_count = 0
        self.interusion_detection_count = 0
        self.denorm_points = [] 
        self.last_alert_sent_time = None
        self.current_insider_bboxes = []
        self.denorm_poly_points = []
        self.is_valid = True
        
        
    

    def __str__(self):
        return f"ROIObject(roi_id={self.roi_id}, name={self.name}, points={self.norm_points}, roi_type={self.roi_type}, created_at={self.created_at}, updated_at={self.updated_at}, update_count={self.update_count})"
      
    def denormalize_points(self, frame_width: int, frame_height: int) -> list:
        """
        Convert normalized points to absolute pixel coordinates based on frame dimensions.
        """
        print(f"Denormalizing points for ROI {self.roi_id=}: {self.norm_points=}, frame_width={frame_width}, frame_height={frame_height}")
        self.denorm_points = [(int(x * frame_width), int(y * frame_height)) for x, y in self.norm_points]
        self.denorm_poly_points = Polygon(self.denorm_points)
        # print(f"Denormalized points for ROI {self.roi_id=}: {self.denorm_points=}")
        return self.denorm_points

    def update_roi(self,roi_data:dict):

        self.points = roi_data.get("points", self.points)
        

