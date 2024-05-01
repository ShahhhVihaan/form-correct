import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_torso_angle(shoulder, elbow, frame_height, frame_width):
    """Calculate the angle between the upper arm and the vertical axis (torso)."""
    shoulder = np.array([shoulder.x * frame_width, shoulder.y * frame_height])
    elbow = np.array([elbow.x * frame_width, elbow.y * frame_height])
    
    # vertical line from elbow, same x, different y
    vertical_point = np.array([elbow[0], elbow[1] - 1])

    shoulder_to_elbow = elbow - shoulder
    elbow_to_vertical = vertical_point - elbow
    cosine_angle = np.dot(shoulder_to_elbow, elbow_to_vertical) / (np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_vertical))
    angle = np.arccos(cosine_angle)  # angle in radians
    angle = np.degrees(angle)  # convert to degrees

    return angle

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

            # Calculate angles
            left_angle = calculate_torso_angle(left_shoulder, left_elbow, frame_height, frame_width)
            right_angle = calculate_torso_angle(right_shoulder, right_elbow, frame_height, frame_width)
            
            angles.append((left_angle, right_angle))

    cap.release()

    print(angles)
    average_angles = np.mean(angles, axis=0)
    left_label = "Good" if 35 <= average_angles[0] <= 55 else "Adjust"
    right_label = "Good" if 35 <= average_angles[1] <= 55 else "Adjust"

    return left_label, right_label

video_path = "Copy of push up 2.mp4"
labels = process_video(video_path)
print("Labels for the video:", labels)