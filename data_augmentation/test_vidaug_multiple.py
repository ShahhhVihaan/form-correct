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

def count_videos(directory):
    return len([name for name in os.listdir(directory) if name.endswith(('.mp4', '.avi'))])

def process_and_save_video(augmented_frames, output_path, fps, frame_size):
    save_video_frames(augmented_frames, output_path, fps, frame_size)

# Maximum number of videos allowed in the output directory
max_videos = 64
input_dir = "/home/vihaan/Projects/form-correct/final_datasets/second_iter/train/1110"
output_dir = "/home/vihaan/Projects/form-correct/final_datasets/second_iter/train/1110"

sometimes = lambda aug: va.Sometimes(1, aug)  # Apply augmentor with 100% probability

# Define a list of different augmentation sequences
augmentation_sequences = [
    va.Sequential([sometimes(va.HorizontalFlip())]),
    va.Sequential([sometimes(va.RandomShear(0.25, 0.1))]),
    va.Sequential([sometimes(va.GaussianBlur(1.2))]),
    va.Sequential([sometimes(va.Salt())]),
    va.Sequential([sometimes(va.Pepper())]),
    va.Sequential([sometimes(va.Add())]),
    va.Sequential([sometimes(va.Multiply())]),
    va.Sequential([sometimes(va.RandomShear(0.25, 0.1))]),
]

filenames = [filename for filename in os.listdir(input_dir)]
# Process each video in the input directory
for filename in filenames:
    if filename.endswith((".mp4", ".avi")) and count_videos(output_dir) < max_videos:
        video_path = os.path.join(input_dir, filename)
        original_frames = load_video_frames(video_path)

        # Get video properties from the original video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()

        # Apply each augmentation sequence to the original video frames
        for i, seq in enumerate(augmentation_sequences):
            if count_videos(output_dir) >= max_videos:
                print(f"Maximum number of videos reached: {max_videos}. Stopping processing.")
                break
            output_video_path = os.path.join(output_dir, f"augmented_{filename[:-4]}_{i+1}.mp4")
            augmented_frames = seq(original_frames)
            process_and_save_video(augmented_frames, output_video_path, fps, frame_size)

    elif count_videos(output_dir) >= max_videos:
        print("Maximum video limit reached. No further processing will be done.")
        break

print("Done processing all videos.")
