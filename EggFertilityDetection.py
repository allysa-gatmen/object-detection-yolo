#!/usr/bin/env python3

import threading
import cv2
import math
import numpy as np
from ultralytics import YOLO
import cvzone
import time
from adafruit_servokit import ServoKit

# Initialize the servo kit
print("SERVO READY")
kit = ServoKit(channels=16)

# Function to rotate the servo
def rotate_servo(index, start_angle, end_angle):
    kit.servo[index].angle = start_angle
    time.sleep(0.5)  # Wait for 0.5 seconds at the rotated position
    kit.servo[index].angle = end_angle
    time.sleep(0.5)  # Wait for 0.5 seconds at the initial position

# Function to rotate servo2
def rotate_servo2(i):
    threading.Thread(target=rotate_servo, args=(i, 180, 0)).start()

# Function to rotate servo1
def rotate_servo1(i):
    threading.Thread(target=rotate_servo, args=(i, 0, 180)).start()

# Initialize the video capture
print("CAMERA INIT")
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # Width
cap.set(4, 1080)  # Height
cv2.namedWindow("Fertility Detection with Marking System", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Fertility Detection with Marking System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
print("CAMERA STARTED")

# Define ROI coordinates
roi_coordinates2 = [
    (0, 910, 925, 1080),  # 1st egg
    (0, 720, 925, 910),   # 2nd egg
    (0, 555, 925, 720),  # 3rd egg
    (0, 375, 925, 555),  # 4th egg
    (0, 190, 925, 375),  # 5th egg
    (0, 0, 925, 190),  # 6th egg
]

roi_coordinates1 = [
    (925, 910, 1920, 1080),  # 7th egg
    (925, 720, 1920, 910),  # 8th egg
    (925, 555, 1920, 720),  # 9th egg
    (925, 375, 1920, 555),  # 10th egg
    (925, 190, 1920, 375),  # 11th egg
    (925, 0, 1920, 190),  # 12th egg
]

# Initialize YOLO model
model = YOLO("/home/egg/Downloads/99.pt")
classNames = ['fertile', 'infertile']

# Main loop
paused = False
last_img = None


def process_and_rotate_servos(infertile_detected_list1, infertile_detected_list2):
    # Rotate servos for infertile detected eggs
    for index, infertile_detected in enumerate(infertile_detected_list1):
        if infertile_detected:
            rotate_servo2(index + 6)

    for index, infertile_detected in enumerate(infertile_detected_list2):
        if infertile_detected:
            rotate_servo1(index)

    # Indicate that the detection and marking process has finished
    print("DONE DETECTING AND MARKING")

    # Resume live detection after servo rotation
    global paused
    paused = False


while True:
    print("RUNNING")
    if not paused:
        success, img = cap.read()
        if not success:
            break
        last_img = img.copy()
    else:
        print("DISOKAY LAST IMAGE")
        if last_img is not None:
            img = last_img.copy()

            # Store infertile detected status for each egg
            infertile_detected_list1 = [False] * len(roi_coordinates1)
            infertile_detected_list2 = [False] * len(roi_coordinates2)

            # Process image detection results for roi_coordinates1
            for index, (roi_x1, roi_y1, roi_x2, roi_y2) in enumerate(roi_coordinates1):
                roi1 = img[roi_y1:roi_y2, roi_x1:roi_x2]
                roi1 = np.ascontiguousarray(roi1)
                # Check if the ROI has valid dimensions
                if roi1.shape[0] == 0 or roi1.shape[1] == 0:
                    continue
                results1 = model(roi1, stream=True)
                # Ensure only one detection (highest confidence) per ROI
                max_confidence = -1
                best_box = None
                for r in results1:
                    for box in r.boxes:
                        conf = box.conf[0]
                        if conf > max_confidence:
                            max_confidence = conf
                            best_box = box
                # Process the best detection (if any)
                if best_box and max_confidence > 0.2:
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                    cls = int(best_box.cls[0])
                    currentClass = classNames[cls]
                    infertile_detected_list1[index] = currentClass == "infertile"
                    print(f"ROI 1 Index: {index}, Detected: {currentClass}, Confidence: {max_confidence:.2f}")  # Debug statement

                    # Calculate adjusted coordinates within the original image
                    x1_original = x1 + roi_x1  # Add ROI's top-left x-coordinate
                    y1_original = y1 + roi_y1  # Add ROI's top-left y-coordinate
                    x2_original = x2 + roi_x1  # Add ROI's top-left x-coordinate
                    y2_original = y2 + roi_y1  # Add ROI's top-left y-coordinate

                    # Draw rectangle, class name, and confidence on the original image
                    myColor = (0, 0, 255) if infertile_detected_list1[index] else (0, 255, 0)
                    cv2.rectangle(img, (x1_original, y1_original), (x2_original, y2_original), myColor, 2)
                    cvzone.putTextRect(img,
                                       f'{currentClass} {max_confidence:.2f}',
                                       (max(0, x1_original), max(35, y1_original)),
                                       scale=1,
                                       thickness=1,
                                       colorB=myColor,
                                       colorT=(255, 255, 255),
                                       cvzone.putTextRect(img,
                                                          f'{currentClass} {max_confidence:.2f}',
                                                          (max(0, x1_original), max(35, y1_original)),
                                                          scale=1,
                                                          thickness=1,
                                                          colorB=myColor,
                                                          colorT=(255, 255, 255),
                                                          colorR=myColor,
                                                          offset=5)

                    # Process image detection results for roi_coordinates2
                    for index, (roi_x1, roi_y1, roi_x2, roi_y2) in enumerate(roi_coordinates2):
                        roi2 = img[roi_y1:roi_y2, roi_x1:roi_x2]
                    roi2 = np.ascontiguousarray(roi2)
                    # Check if the ROI has valid dimensions
                    if roi2.shape[0] == 0 or roi2.shape[1] == 0:
                        continue
                    results2 = model(roi2, stream=True)
                    # Ensure only one detection (highest confidence) per ROI
                    max_confidence = -1
                    best_box = None
                    for r in results2:
                        for box in r.boxes:
                            conf = box.conf[0]
                            if conf > max_confidence:
                                max_confidence = conf
                                best_box = box
                    # Process the best detection (if any)
                    if best_box and max_confidence > 0.2:
                        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                        cls = int(best_box.cls[0])
                        currentClass = classNames[cls]
                        infertile_detected_list2[index] = currentClass == "infertile"
                        print(
                            f"ROI 2 Index: {index}, Detected: {currentClass}, Confidence: {max_confidence:.2f}")  # Debug statement

                        # Calculate adjusted coordinates within the original image
                        x1_original = x1 + roi_x1  # Add ROI's top-left x-coordinate
                        y1_original = y1 + roi_y1  # Add ROI's top-left y-coordinate
                        x2_original = x2 + roi_x1  # Add ROI's top-left x-coordinate
                        y2_original = y2 + roi_y1  # Add ROI's top-left y-coordinate

                        # Draw rectangle, class name, and confidence on the original image
                        myColor = (0, 0, 255) if infertile_detected_list2[index] else (0, 255, 0)
                        cv2.rectangle(img, (x1_original, y1_original), (x2_original, y2_original), myColor, 2)
                    cvzone.putTextRect(img,
                                       f'{currentClass} {max_confidence:.2f}',
                                       (max(0, x1_original), max(35, y1_original)),
                                       scale=1,
                                       thickness=1,
                                       colorB=myColor,
                                       colorT=(255, 255, 255),
                                       colorR=myColor,
                                       offset=5)

            # Process and rotate servos in a separate thread
            threading.Thread(target=process_and_rotate_servos, args=(infertile_detected_list1, infertile_detected_list2)).start()

    cv2.imshow("Fertility Detection with Marking System", img)
    key = cv2.waitKey(1)  # Waits for 1 ms for a key press
    if key == 27:  # If ESC is pressed, exit loop
        break
    elif key == 32:  # If SPACEBAR is pressed, toggle pause state
        paused = not paused

cap.release()
cv2.destroyAllWindows()
print("endcam")

