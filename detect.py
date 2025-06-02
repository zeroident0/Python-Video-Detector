from ultralytics import YOLO
import cv2
import numpy as np


def apply_night_vision(frame):
    """Apply night vision effects to the frame"""
    # Convert to grayscale for night vision effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to enhance contrast
    gray_eq = cv2.equalizeHist(gray)

    # Create pseudo-green night vision effect
    night_vision = cv2.applyColorMap(gray_eq, cv2.COLORMAP_JET)

    # Blend with original image for better visibility
    alpha = 0.7  # Weight for night vision effect
    beta = 0.3  # Weight for original image
    blended = cv2.addWeighted(frame, beta, night_vision, alpha, 0)

    return blended


# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 nano model

# Load a video
video_path = 'cars.mp4'
cap = cv2.VideoCapture(video_path)

# Create toggle for night vision mode
night_vision_mode = False

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no more frames

    # Apply night vision if enabled
    if night_vision_mode:
        display_frame = apply_night_vision(frame)
    else:
        display_frame = frame.copy()

    # Run YOLOv8 detection on original frame (not night vision processed)
    results = model(frame)

    # Draw bounding boxes and labels on the display frame
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Get class ID and name
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            # Get confidence score
            conf = float(box.conf[0])

            # Choose color based on night vision mode
            box_color = (0, 255, 0) if night_vision_mode else (255, 0, 0)  # Green for night, blue for day
            text_bg_color = (0, 100, 0) if night_vision_mode else (255, 0, 0)  # Dark green for night, blue for day

            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

            # Create label text
            label = f"{cls_name} {conf:.2f}"

            # Calculate text size
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw background rectangle for text
            cv2.rectangle(display_frame, (x1, y1 - text_height - 5),
                          (x1 + text_width, y1), text_bg_color, -1)

            # Put text on image
            text_color = (255, 255, 255)  # White text for both modes
            cv2.putText(display_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Display mode information
    mode_text = "Night Vision ON" if night_vision_mode else "Normal Mode"
    cv2.putText(display_frame, mode_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if night_vision_mode else (255, 0, 0), 2)

    # Display the frame with detections
    cv2.imshow('YOLOv8 Object Detection', display_frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):  # Toggle night vision mode
        night_vision_mode = not night_vision_mode

# Release resources
cap.release()
cv2.destroyAllWindows()