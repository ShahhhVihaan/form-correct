import cv2
import os
import numpy as np
import vidaug.augmentors as va

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video_frames(frames, output_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for frame in frames:
        out.write(frame)
    out.release()
    
input_dir = "/home/vihaan/Projects/form-correct/videos/input/"
output_dir = "/home/vihaan/Projects/form-correct/videos/augmented/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# sometimes = lambda aug: va.Sometimes(1, aug)  # Apply augmentor with 100% probability
# seq = va.Sequential([
#     sometimes(va.HorizontalFlip()),  # Horizontally flip the video with 100% probability
# ])
sometimes = lambda aug: va.Sometimes(1, aug)  # 50% probability to better illustrate variety

seq = va.Sequential([
    # sometimes(va.HorizontalFlip()),  # Horizontally flip the video
    # sometimes(va.RandomRotate(degrees=25)),  # Rotate the video by 10 degrees
    # sometimes(va.GaussianBlur(1.5)),  # Apply Gaussian Blur with a sigma of 1.5
    # sometimes(va.Add(-100)),  # Increase brightness
    # sometimes(va.Multiply(0.9)),  # Decrease brightness by multiplying by 0.9
    sometimes(va.RandomShear(0.25, 0.1)),  # Add salt and pepper noise with 5% probability per pixel
])

# Process each video in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith((".mp4", ".avi")):  # Check for video files
        video_path = os.path.join(input_dir, filename)
        frames = load_video_frames(video_path)
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        
        # Apply augmentations
        augmented_frames = seq(frames)
        
        # Save augmented video
        output_path = os.path.join(output_dir, "aug_" + filename)
        save_video_frames(augmented_frames, output_path, fps, frame_size)

print("Done processing all videos.")
