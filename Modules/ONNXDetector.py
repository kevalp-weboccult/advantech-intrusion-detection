from __future__ import annotations
import time
import onnxruntime as ort
import cv2
import numpy as np
import os 
from Modules.CustomLogger import CustomLogger
from managers.ConfigManager import ConfigManager
from typing import List,Dict,Any,Optional
import traceback 
from utils.bbox_utils import multiclass_nms
class ONNXDetector:
    

    def __init__(self,model_path):
        """
        Initialize an instance of the YOLOv8 class.

        Args:
            onnx_model (str): Path to the ONNX model.
            input_image (str): Path to the input image.
            confidence_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IoU threshold for non-maximum suppression.
        """
        # self.onnx_model = onnx_model
        self.config_manager = ConfigManager.get_instance()
        self.logger = CustomLogger("YOLOv8").get_logger(
            log_file="logs/YOLOv8.log",
            log_level=self.config_manager.get("LOG_LEVEL"),
            log_to_console=self.config_manager.get("LOG_TO_CONSOLE"),
        )
        self.model_path:str = model_path #os.path.join(self.config_manager.get("Model_folder"),self.config_manager.get("DETECTION_MODEL_PATH","yolov8s.onnx"))
        self.providers:List[str] = self.config_manager.get("ONNX_PROVIDER",["CUDAExecutionProvider","CPUExecutionProvider"])
        
        try:
            self.session = ort.InferenceSession(self.model_path, providers=self.providers)
            self.logger.info(f"Model loaded successfully from {self.model_path} using providers: {self.providers}")
        except Exception as e:
            self.logger.error(f"Error loading model from {self.model_path}: {e}")
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            self.logger.info(f"Fallback to CPUExecutionProvider for model loading from {self.model_path}")
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.confidence_thres:float = self.config_manager.get("DETECTION_MODEL_CONF",0.25)
        self.iou_thres:float = self.config_manager.get("DETECTION_MODEL_IOU",0.7)
        self.catrogry_to_detect:List[str] = self.config_manager.get("CATEGORY_TO_DETECT",["body"])
        
        
            

        # Load the class names from the COCO dataset
        self.classes:Dict[int,str] = {0:"body",1:"head"}

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self._warmup()
        
    
    def _warmup(self)->None:
        """
        Perform a warm-up inference to optimize model performance.

        This method runs a dummy inference using a zeroed input tensor to
        initialize the model and optimize it for subsequent real inferences.
        """
        dummy_input = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)
        try:
            self.session.run(None, {self.input_name: dummy_input})
            self.logger.info("Model warm-up completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during model warm-up: {e}")
            self.logger.error(traceback.format_exc())

    def letterbox(self, img: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            img (np.ndarray): Resized and padded image.
            pad (tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

    def draw_detections(self, img: np.ndarray, box: list[float], score: float, class_id: int) -> None:
        """Draw bounding boxes and labels on the input image based on the detected objects."""
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    def preprocess(self,img,input_size) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Preprocess the input image before performing inference.

        This method reads the input image, converts its color space, applies letterboxing to maintain aspect ratio,
        normalizes pixel values, and prepares the image data for model input.

        Returns:
            image_data (np.ndarray): Preprocessed image data ready for inference with shape (1, 3, height, width).
            pad (tuple[int, int]): Padding values (top, left) applied during letterboxing.
        """
        # Read the input image using OpenCV


        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_height, self.img_width = img.shape[:2]

        img, pad = self.letterbox(img, (input_size, input_size))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data, pad

    def postprocess(self, input_image: np.ndarray, output: list[np.ndarray], pad: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Efficiently process model outputs to extract bounding boxes, scores, and class IDs using vectorized operations.

        Args:
            input_image (np.ndarray): The input image.
            output (list[np.ndarray]): The raw output from the model.
            pad (tuple[int, int]): Padding (top, left) used during preprocessing.

        Returns:
            tuple: (boxes, scores, class_ids) filtered by confidence and NMS.
        """
        outputs = np.transpose(np.squeeze(output[0]))  # Shape: (N, num_attrs)
        
        # Remove padding
        outputs[:, 0] -= pad[1]  # x
        outputs[:, 1] -= pad[0]  # y

        # Scale coordinates
        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)

        # Extract class scores, best score, and class IDs in a vectorized way
        class_scores = outputs[:, 4:]  # shape: (N, num_classes)
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence threshold
        conf_mask = max_scores >= self.confidence_thres
        outputs = outputs[conf_mask]
        max_scores = max_scores[conf_mask]
        class_ids = class_ids[conf_mask]

        if outputs.shape[0] == 0:
            print("output shape is zero ")
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

        # Convert class names to IDs only once
        # allowed_class_ids = {
        #     self.classes.index(cls_name)
        #     for cls_name in self.catrogry_to_detect if cls_name in self.classes
        # }
        index_to_consider = []
        for idx,cls in enumerate(class_ids):
            if self.classes[cls] in self.catrogry_to_detect:
                index_to_consider.append(idx)
            
        print(index_to_consider)
        index_to_consider = np.array(index_to_consider)
        print(index_to_consider)
        # exit()


        # print("allowd",allowed_class_ids)

        # Filter by allowed class IDs
        # allowed_mask = np.array([cls in allowed_class_ids for cls in class_ids])
        if index_to_consider.shape[0]==0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
        outputs = outputs[index_to_consider]
        max_scores = max_scores[index_to_consider]
        class_ids = class_ids[index_to_consider]

        if outputs.shape[0] == 0:
            print("no allowed class detected")
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

        # Convert center x, y, width, height to x1, y1, x2, y2
        x = outputs[:, 0]
        y = outputs[:, 1]
        w = outputs[:, 2]
        h = outputs[:, 3]

        x = (x - w / 2) / gain
        y = (y - h / 2) / gain
        w = (w) / gain
        h = (h) / gain

        boxes = np.stack([x, y, x+w, y+h], axis=1).astype(np.int32)

        # Apply NMS
        s_time = time.time()
        indices = multiclass_nms(boxes, max_scores, class_ids, iou_threshold=self.iou_thres)
        e_time = time.time()
        print("Time taken in NMS:", e_time - s_time)

        # Final selection
        boxes = boxes[indices]
        scores = max_scores[indices]
        class_ids = class_ids[indices]

        # print(f"{boxes.shape=} {scores.shape=} {class_ids.shape=}")

        return boxes, scores, class_ids

    def _postprocess(self, input_image: np.ndarray, output: list[np.ndarray], pad: tuple[int, int]) -> np.ndarray:
        #TODO: Imrpove the postprocessing function to take less time.
        """
        Perform post-processing on the model's output to extract and visualize detections.

        This method processes the raw model output to extract bounding boxes, scores, and class IDs.
        It applies non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            input_image (np.ndarray): The input image.
            output (list[np.ndarray]): The output arrays from the model.
            pad (tuple[int, int]): Padding values (top, left) used during letterboxing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)

                # Add the class ID, score, and box coordinates to the respective lists
                
                if self.classes[class_id] not in self.catrogry_to_detect:
                    # print(f"Skipping class: {self.classes[class_id]}")
                    continue
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width+left, height+top])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        # indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        # indices = np.array(indices).flatten() if len(indices) > 0 else []
        s_time = time.time()
        indices = multiclass_nms(boxes,scores,class_ids,iou_threshold=self.iou_thres)
        e_time = time.time()
        print(f"Time taken in NMS",e_time-s_time)
        #select the indices values from the boxes, scores and class_ids
        boxes = boxes[indices]
        scores = scores[indices]
        class_ids = class_ids[indices]
        # Iterate over the selected indices after non-maximum suppression
        # for i in indices:
        #     # Get the box, score, and class ID corresponding to the index
        #     box = boxes[i]
        #     score = scores[i]
        #     class_id = class_ids[i]

            # Draw the detection on the input image
            # self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return boxes, scores, class_ids

    def detect(self, image: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Perform inference on the input image and return the detected bounding boxes, scores, and class IDs.

        This method preprocesses the input image, runs the model inference, and post-processes the output
        to extract and visualize detections.

        Args:
            image (np.ndarray): The input image for inference.
        """
        
        try:
            # Preprocess the input image
            s_time = time.time_ns()
            input_data, pad = self.preprocess(image,self.input_width)
            e_time = time.time_ns()
            print(f"Preprocess time: {(e_time - s_time)/1e6} ms",flush=True)

            # Run the model inference
            s_time = time.time_ns()
            output = self.session.run(None, {self.input_name: input_data})
            e_time = time.time_ns()
            print(f"Inference time: {(e_time - s_time)/1e6} ms",flush=True)

            # Post-process the model output to extract and visualize detections
            s_time = time.time_ns()
            boxes, scores, class_ids = self.postprocess(image, output, pad)
            e_time = time.time_ns()
            print(f"Postprocess time: {(e_time - s_time)/1e6} ms",flush=True)

            return boxes, scores, class_ids
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            self.logger.error(traceback.format_exc())
            return [],[],[]

if __name__ == "__main__":
    yolo = ONNXDetector()
    yolo._warmup()
    image = cv2.imread("test_2.jpg")
    result = yolo.detect(image)
    if result:
        boxes, scores, class_ids = result
        for box, score, class_id in zip(boxes, scores, class_ids):
            # plotted_image = yolo.draw_detections(image, box, score, class_id)
            cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,0),1)

        cv2.imshow("img",image)
        cv2.waitKey(0)