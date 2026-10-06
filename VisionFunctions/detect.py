import cv2
import numpy as np

CONTRAST = 1.1
BRIGHTNESS = 1
BLUR_SIZE = 1
THRESHOLD_VALUE = 254
MIN_CONTOUR_AREA = 100


def detect_objects(image, contrast = CONTRAST, brightness=BRIGHTNESS, blur_size=BLUR_SIZE, threshold_value=THRESHOLD_VALUE, min_contour_area=MIN_CONTOUR_AREA):
    if image is None:
        raise ValueError("Image not found or unable to read the image.")

    contrasted_image = cv2.convertScaleAbs(image, alpha = contrast, beta = brightness)
    grayscale_image = cv2.cvtColor(contrasted_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (blur_size, blur_size), 0)
    _, threshold = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("Image 1", threshold)
    cv2.imshow("Image 2", image)
    cv2.imshow("Image 3", cv2.bitwise_and(image, image, mask=threshold))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    detected_objects = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue

        rectangle = cv2.minAreaRect(contour)

        (center_x, center_y), (w, h), angle = rectangle

        if w < h:
            angle += 90

        center_x, center_y = round(center_x, 1), round(center_y, 1)
        angle = round(angle, 1)

        detected_objects.append({
            "center_x": center_x,
            "center_y": center_y,
            "angle": angle,
            "area": area,
            "contour": contour,
            "bounding_box": rectangle
        })

    return detected_objects