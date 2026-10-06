from cv2 import imread, imwrite, boxPoints, drawContours, circle, putText, FONT_HERSHEY_SIMPLEX
import numpy as np
from VisionFunctions.detect import detect_objects
from VisionFunctions.transform import transform_objects
from VisionFunctions.classify import classify_objects

image_path = "ExampleData/Vision example5.webp"
image = imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

new_image = image.copy()
detected_objects = detect_objects(image)

transformed_objects = transform_objects(image, detected_objects)

cluster_labels = classify_objects(
    transformed_objects,
    eps=0.83
    #28,
    #1
)

for index, obj in enumerate(detected_objects):
    box = boxPoints(obj["bounding_box"])
    box = np.intp(box)

    label = str(cluster_labels[index]) if index < len(cluster_labels) else "?"

    drawContours(new_image, [box], 0, (0, 150, 255), 2)
    circle(new_image, (int(obj["center_x"]), int(obj["center_y"])), 5, (0, 0, 255), -1)
    putText(new_image, label, (int(obj["center_x"]), int(obj["center_y"]) - 10),
    FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

imwrite("ExampleData/Vision example5 detected.webp", new_image)
print(f"Detected {len(detected_objects)} objects with labels: {cluster_labels}")
