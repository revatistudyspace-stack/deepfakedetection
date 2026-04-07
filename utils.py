import cv2

def extract_frames(video_path, frame_size=(224, 224), sample_rate=5):
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Take every Nth frame (optimization)
        if count % sample_rate == 0:
            frame = cv2.resize(frame, frame_size)
            frames.append(frame)

        count += 1

    cap.release()
    return frames