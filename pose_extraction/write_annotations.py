import cv2
import mediapipe as mp
import json
import os

def process_video(video_path, output_video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        smooth_landmarks=True,
                        enable_segmentation=False,
                        smooth_segmentation=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    pose_data = {}

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        frame_annotated = frame.copy()

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame_annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            landmarks = [{'x': landmark.x, 'y': landmark.y, 'z': landmark.z, 'visibility': landmark.visibility}
                         for landmark in results.pose_landmarks.landmark]
            pose_data[frame_count] = landmarks

        out.write(frame_annotated)
        frame_count += 1

    cap.release()
    out.release()

    json_filename = os.path.splitext(video_path)[0] + ".json"
    with open(json_filename, 'w') as f:
        json.dump(pose_data, f, indent=4)

    return json_filename

video_path = "Copy of push up 2.mp4"
output_video_path = "annotated_video.mp4"
json_filename = process_video(video_path, output_video_path)

print(f"Pose data saved to {json_filename}")

def play_pose_annotations(video_path, json_path):
    cap = cv2.VideoCapture(video_path)
    with open(json_path, 'r') as f:
        pose_data = json.load(f)

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success or str(frame_count) not in pose_data:
            break

        frame_annotated = frame.copy()
        landmarks = pose_data[str(frame_count)]
        for landmark in landmarks:
            x = int(landmark['x'] * frame.shape[1])
            y = int(landmark['y'] * frame.shape[0])
            cv2.circle(frame_annotated, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow('Pose Playback', frame_annotated)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

play_pose_annotations(video_path, json_filename)
