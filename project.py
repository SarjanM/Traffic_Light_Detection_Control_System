import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # You can also use yolov8.pt if available

# Define HSV color ranges for red, yellow, and green
color_ranges = {
    "red": [(0, 100, 100), (10, 255, 255)],
    "yellow": [(15, 100, 100), (35, 255, 255)],
    "green": [(40, 50, 50), (90, 255, 255)],
}

def detect_traffic_light_color(roi):
    # Convert the region of interest (ROI) to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Check for each color in the defined ranges
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        color_pixels = cv2.countNonZero(mask)
        if color_pixels > 30:  # Threshold to consider color detected
            return color_name
    return "unknown"

cap = cv2.VideoCapture("C:/Users/serca/Desktop/4th_course/Image_Processing/IP_project/video2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame)

    # Initialize counters for each color
    color_counts = {"red": 0, "yellow": 0, "green": 0, "unknown": 0}

    # Process each detected object
    for result in results[0].boxes:
        # Check if the detected object is a traffic light (COCO class ID for traffic light is 9 in YOLO)
        if int(result.cls[0]) == 9:  # Class ID for "traffic light"
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates

            # Extract the region of interest (ROI) for color detection
            roi = frame[y1:y2, x1:x2]

            # Detect the color of the traffic light
            color = detect_traffic_light_color(roi)
            color_counts[color] += 1  # Increment the count for the detected color
            
            # Define color mappings for bounding boxes
            color_map = {
                "red": (0, 0, 255),
                "yellow": (0, 255, 255),
                "green": (0, 255, 0),
                "unknown": (255, 0, 0),
            }
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[color], 2)
            cv2.putText(frame, f"Traffic Light - {color}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[color], 2)

    # Draw the information box on the right corner of the screen
    info_box_start = (480, 500)
    info_box_end = (630, 350)
    cv2.rectangle(frame, info_box_start, info_box_end, (255, 255, 255), -1)
    cv2.putText(frame, "Traffic Lights", (490, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    y_offset = 410
    for color, count in color_counts.items():
        cv2.putText(frame, f"{color.capitalize()}: {count}", (490, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        y_offset += 20

    # Display the frame with detections
    cv2.imshow("Traffic Light Detection with Color", frame)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
