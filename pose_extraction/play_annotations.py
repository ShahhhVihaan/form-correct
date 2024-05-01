import cv2
import json
import numpy as np
import mediapipe as mp

def play_pose_annotations(video_path, json_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    
    with open(json_path, 'r') as f:
        pose_data = json.load(f)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # get current frame index
        if str(frame_count) in pose_data:
            landmarks_data = pose_data[str(frame_count)]

            for landmark in landmarks_data:
                x = int(landmark['x'] * frame.shape[1])
                y = int(landmark['y'] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (245, 117, 66), -1)

            if len(landmarks_data) > 16: 
                lmk1 = landmarks_data[0]
                lmk2 = landmarks_data[1]
                x1, y1 = int(lmk1['x'] * frame.shape[1]), int(lmk1['y'] * frame.shape[0])
                x2, y2 = int(lmk2['x'] * frame.shape[1]), int(lmk2['y'] * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (245, 66, 230), 2)

        cv2.imshow('Pose Playback', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_path = "Copy of push up 2.mp4"
json_path = "Copy of push up 2.json"
play_pose_annotations(video_path, json_path)
