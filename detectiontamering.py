import os
import cv2
import numpy as np
from VisionFunctions.detect import detect_objects

image_path = "ExampleData/Vision example1.webp"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
for i in range(1):
    print(i)
    detected_objects = detect_objects(
        image, 
        contrast =1.1,
        brightness=10,
        blur_size=1,
        threshold_value=254,
        min_contour_area=100
    )