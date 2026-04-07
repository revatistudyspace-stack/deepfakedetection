import os
import cv2
import numpy as np
from utils import extract_frames

video_path = "samples/sample_video.mp4"

os.makedirs(os.path.dirname(video_path), exist_ok=True)

if not os.path.exists(video_path):
    print("Generating local sample video...")
    width, height = 320, 240
    fps = 10
    frame_count = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        color = (int(255 * (i / frame_count)), 128, int(255 * (1 - i / frame_count)))
        cv2.rectangle(frame, (20 + i * 5, 40), (120 + i * 5, 140), color, -1)
        cv2.putText(frame, f"Frame {i+1}", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    print("Sample video created at:", video_path)

frames = extract_frames(video_path)
print("Total frames extracted:", len(frames))