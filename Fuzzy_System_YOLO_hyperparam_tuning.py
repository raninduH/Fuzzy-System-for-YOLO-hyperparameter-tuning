import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skimage import exposure, filters, util
from skimage.restoration import estimate_sigma

# Define fuzzy input variables
brightness = ctrl.Antecedent(np.arange(0, 11, 1), 'brightness')
contrast = ctrl.Antecedent(np.arange(0, 11, 1), 'contrast')
noise_level = ctrl.Antecedent(np.arange(0, 11, 1), 'noise_level')
sharpness = ctrl.Antecedent(np.arange(0, 11, 1), 'sharpness')
object_distance = ctrl.Antecedent(np.arange(0, 11, 1), 'object_distance')
resolution_quality = ctrl.Antecedent(np.arange(0, 11, 1), 'resolution_quality')

# Define fuzzy output variables for YOLO parameters
conf_thres = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'conf_thres')
iou_thres = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'iou_thres')
img_size = ctrl.Consequent(np.arange(320, 1281, 160), 'img_size')
max_detections = ctrl.Consequent(np.arange(0, 1001, 50), 'max_detections')

# Define membership functions for inputs
brightness['low'] = fuzz.trapmf(brightness.universe, [0, 0, 2, 4])
brightness['medium'] = fuzz.trimf(brightness.universe, [3, 5, 7])
brightness['high'] = fuzz.trapmf(brightness.universe, [6, 8, 10, 10])

contrast['low'] = fuzz.trapmf(contrast.universe, [0, 0, 2, 4])
contrast['medium'] = fuzz.trimf(contrast.universe, [3, 5, 7])
contrast['high'] = fuzz.trapmf(contrast.universe, [6, 8, 10, 10])

noise_level['low'] = fuzz.trapmf(noise_level.universe, [0, 0, 2, 4])
noise_level['medium'] = fuzz.trimf(noise_level.universe, [3, 5, 7])
noise_level['high'] = fuzz.trapmf(noise_level.universe, [6, 8, 10, 10])

sharpness['low'] = fuzz.trapmf(sharpness.universe, [0, 0, 2, 4])
sharpness['medium'] = fuzz.trimf(sharpness.universe, [3, 5, 7])
sharpness['high'] = fuzz.trapmf(sharpness.universe, [6, 8, 10, 10])

object_distance['close'] = fuzz.trapmf(object_distance.universe, [0, 0, 2, 4])
object_distance['medium'] = fuzz.trimf(object_distance.universe, [3, 5, 7])
object_distance['far'] = fuzz.trapmf(object_distance.universe, [6, 8, 10, 10])

resolution_quality['low'] = fuzz.trapmf(resolution_quality.universe, [0, 0, 2, 4])
resolution_quality['medium'] = fuzz.trimf(resolution_quality.universe, [3, 5, 7])
resolution_quality['high'] = fuzz.trapmf(resolution_quality.universe, [6, 8, 10, 10])

# Define fuzzy membership functions for outputs
conf_thres['low'] = fuzz.trapmf(conf_thres.universe, [0, 0, 0.2, 0.4])
conf_thres['medium'] = fuzz.trimf(conf_thres.universe, [0.3, 0.5, 0.7])
conf_thres['high'] = fuzz.trapmf(conf_thres.universe, [0.6, 0.8, 1, 1])

iou_thres['low'] = fuzz.trapmf(iou_thres.universe, [0, 0, 0.2, 0.4])
iou_thres['medium'] = fuzz.trimf(iou_thres.universe, [0.3, 0.5, 0.7])
iou_thres['high'] = fuzz.trapmf(iou_thres.universe, [0.6, 0.8, 1, 1])

img_size['small'] = fuzz.trapmf(img_size.universe, [320, 320, 480, 640])
img_size['medium'] = fuzz.trimf(img_size.universe, [480, 640, 800])
img_size['large'] = fuzz.trapmf(img_size.universe, [800, 960, 1280, 1280])

max_detections['low'] = fuzz.trapmf(max_detections.universe, [0, 0, 200, 400])
max_detections['medium'] = fuzz.trimf(max_detections.universe, [300, 500, 700])
max_detections['high'] = fuzz.trapmf(max_detections.universe, [600, 800, 1000, 1000])

# Define rules (based on previous example)
rules = [
    ctrl.Rule(brightness['low'] | contrast['low'], (conf_thres['low'], iou_thres['low'], img_size['small'], max_detections['medium'])),
    ctrl.Rule(brightness['high'], (conf_thres['low'], iou_thres['low'], img_size['small'])),
    ctrl.Rule(brightness['medium'] & contrast['high'], (conf_thres['high'], iou_thres['high'], img_size['large'])),
    ctrl.Rule(noise_level['high'], (conf_thres['low'], iou_thres['high'], img_size['small'], max_detections['low'])),
    ctrl.Rule(sharpness['high'], (conf_thres['high'], iou_thres['high'], img_size['large'])),
    ctrl.Rule(object_distance['far'], (conf_thres['low'], img_size['large'], max_detections['high'])),
    ctrl.Rule(object_distance['close'], (img_size['small'], max_detections['medium'])),
    ctrl.Rule(resolution_quality['low'], (conf_thres['low'], iou_thres['low'], img_size['small'], max_detections['low'])),
    ctrl.Rule(resolution_quality['high'], (conf_thres['high'], iou_thres['high'], img_size['large'], max_detections['high']))
]

# Create fuzzy control system
yolo_ctrl = ctrl.ControlSystem(rules)
yolo_simulation = ctrl.ControlSystemSimulation(yolo_ctrl)

def calculate_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray) / 255.0 * 10
    return brightness

def calculate_contrast(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray) / 255.0 * 10
    return contrast

def calculate_noise_level(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    noise_level = estimate_sigma(gray) / 255.0 * 10
    return noise_level

def calculate_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 100.0
    return min(sharpness, 10)

def calculate_object_distance(frame):
    object_distance = 5  # Example value; this can be dynamically calculated with depth info
    return object_distance

def calculate_resolution_quality(frame):
    height, width = frame.shape[:2]
    resolution_quality = min(height * width / (1280 * 720), 10)
    return resolution_quality


# Install yt-dlp if not already installed

import yt_dlp

# Define the YouTube video URL
video_url = 'https://youtu.be/BVH8EZFuXYA'

# Set options for yt-dlp
ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',  # Best video and audio
    'outtmpl': 'downloaded_video.mp4',  # Output file name
    'merge_output_format': 'mp4',       # Ensure the output is in mp4 format
    'quiet': True                       # Suppress output
}

# Download the video
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

print("Video downloaded as 'downloaded_video.mp4'")

import cv2
import torch
from google.colab.patches import cv2_imshow

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()
print('YOLOv5 model loaded successfully.')

# Load the downloaded video
video_path = 'downloaded_video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
else:
    # Retrieve frame width, height, and FPS for the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 FPS if unavailable

    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_with_detections.mp4', fourcc, fps, (frame_width, frame_height))
    print(f"Video writer initialized with dimensions: {frame_width}x{frame_height}, FPS: {fps}")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read, ending video processing.")
            break

        frame_count += 1

        # Ensure the frame is valid
        if frame is None:
            print("Warning: Empty frame encountered, skipping.")
            continue

        # Display progress
        print(f"Processing frame {frame_count}")

        # Example calculations for environmental metrics (assuming these functions are defined)
        brightness_val = calculate_brightness(frame)
        contrast_val = calculate_contrast(frame)
        noise_level_val = calculate_noise_level(frame)
        sharpness_val = calculate_sharpness(frame)
        object_distance_val = calculate_object_distance(frame)
        resolution_quality_val = calculate_resolution_quality(frame)

        # Set fuzzy inputs
        yolo_simulation.input['brightness'] = brightness_val
        yolo_simulation.input['contrast'] = contrast_val
        yolo_simulation.input['noise_level'] = noise_level_val
        yolo_simulation.input['sharpness'] = sharpness_val
        yolo_simulation.input['object_distance'] = object_distance_val
        yolo_simulation.input['resolution_quality'] = resolution_quality_val

        # Compute fuzzy logic output
        yolo_simulation.compute()

        # Get adjusted YOLO parameters from fuzzy logic
        conf_thres_val = yolo_simulation.output['conf_thres']
        iou_thres_val = yolo_simulation.output['iou_thres']

        # Apply fuzzy logic adjustments to YOLO model parameters
        model.conf = conf_thres_val  # Set confidence threshold
        model.iou = iou_thres_val    # Set IoU threshold

        # Run YOLO model on the frame
        results = model(frame)

        # Extract bounding boxes and labels
        detections = results.pandas().xyxy[0]
        for _, row in detections.iterrows():
            # Extract box coordinates and label information
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            confidence = row['confidence']
            class_name = row['name']

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label and confidence
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Overlay environmental metrics and YOLO parameters on the frame
        cv2.putText(frame, f'Brightness: {brightness_val:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'Contrast: {contrast_val:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'Noise Level: {noise_level_val:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'Sharpness: {sharpness_val:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'Resolution Quality: {resolution_quality_val:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'Confidence Threshold: {conf_thres_val:.2f}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'IoU Threshold: {iou_thres_val:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Video processing completed. Saved as 'output_with_detections.mp4' with {frame_count} frames.")

