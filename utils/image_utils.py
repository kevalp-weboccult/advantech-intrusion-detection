"""
Image processing utilities for the Advantech Intrusion Detection System.

This module provides utility functions for image preprocessing operations
including rotation and resizing of video frames from camera streams.
"""

import cv2
import numpy as np
from numpy.typing import NDArray
import traceback


def preprocess_frame(
    frame: NDArray[np.uint8],
    rotation: int,
    resize: bool,
    width: int,
    height: int
) -> NDArray[np.uint8]:
    """
    Preprocess a video frame by rotating and resizing it if necessary.

    This function applies rotation and resizing transformations to a video frame
    based on the provided parameters. It supports 90, 180, and 270-degree rotations
    and optional resizing to specified dimensions.

    Args:
        frame: The input video frame as a NumPy array.
        rotation: The rotation angle in degrees. Supported values are:
                 -1 (no rotation), 90, 180, 270.
        resize: Whether to resize the frame to specified dimensions.
        width: The target width for resizing (used only if resize is True).
        height: The target height for resizing (used only if resize is True).

    Returns:
        The processed frame as a NumPy array with the same dtype as input.

    Example:
        >>> import numpy as np
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> processed = preprocess_frame(frame, 90, True, 320, 240)
        >>> processed.shape
        (240, 320, 3)

    Note:
        - Rotation is applied before resizing if both operations are requested.
        - Invalid rotation values (not -1, 90, 180, or 270) are ignored.
    """
    processed_frame = frame.copy()

    # Apply rotation if specified
    if rotation != -1:
        if rotation == 90:
            processed_frame = cv2.rotate(processed_frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            processed_frame = cv2.rotate(processed_frame, cv2.ROTATE_180)
        elif rotation == 270:
            processed_frame = cv2.rotate(
                processed_frame, cv2.ROTATE_90_COUNTERCLOCKWISE
            )

    # Apply resizing if requested
    if resize:
        processed_frame = cv2.resize(processed_frame, (width, height))

    return processed_frame

def get_black_out_frame(frame,area_check_roi):
    
    try:
        if area_check_roi:
            blank_dummy_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
            # blank_dummy_image[self.area_check_roi[1]:self.area_check_roi[3],self.area_check_roi[0]:self.area_check_roi[2]] = frame[self.area_check_roi[1]:self.area_check_roi[3],self.area_check_roi[0]:self.area_check_roi[2]]
            
            blank_dummy_image = cv2.fillPoly(blank_dummy_image, [area_check_roi], (255, 255, 255))
            blank_dummy_image = cv2.bitwise_and(frame, blank_dummy_image)
            return blank_dummy_image
        else:
            return frame
    except Exception as exec:
        print(f"Error in get_black_out_frame: {exec} {traceback.format_exc()}")  
        return frame    

def get_in_range_polygon(roi_points):
        try:
            for i,point in enumerate(roi_points):
                roi_points[i] = [max(0,min(1,point[0])),max(0,min(1,point[1]))]
            return roi_points
        except Exception as exec:
            print(f"Error in get_in_range_polygon: {exec} {traceback.format_exc()}")  
            return []
        
def denormalize_polygon(roi_points_original,frame_width,frame_height):
        try:
            roi_points = roi_points_original.copy()
            roi_points = get_in_range_polygon(roi_points)
            for i,point in enumerate(roi_points):
                roi_points[i] = [int(point[0] * frame_width),int(point[1] * frame_height)]
            return roi_points
        except Exception as exec:
            print(f"Error in denormalize_polygon: {exec} {traceback.format_exc()}")  
            return []   

def create_blackout_image(image,roi_objects):
    try:
        original_image = image.copy()
        black_out_frame = image.copy()
        for roi_object in roi_objects:
            if roi_object is None or not roi_object.denorm_points:
                continue
            mask_image = np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
            cv2.fillPoly(mask_image, [np.array(roi_object.denorm_points,dtype=np.int32)], 255)
            black_out_frame = cv2.bitwise_and(image, image, mask=mask_image)
            # cv2.imwrite(f"debug_{roi_object.roi_id}.png",image)
        return black_out_frame,original_image



    
    except Exception as exec:
        print(f"Error in create_blackout_image: {exec} {traceback.format_exc()}")  
        return image,image
