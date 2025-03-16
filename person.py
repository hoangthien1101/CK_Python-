import cv2
import torch
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

cv2.setUseOptimized(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

yolo_model = YOLO(r"D:\WorkSpace\Homework\NAM3\HK2\LT_Python\person.pt").to(device)
tracker = DeepSort(max_age=30, n_init=3, nn_budget=20)


def detect_and_track_webcam():
    cap = cv2.VideoCapture(r"D:\WorkSpace\Homework\NAM3\HK2\LT_Python\Person.mp4")


    cap.set(cv2.CAP_PROP_FPS, 15)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width, new_height = 480, 270

    if not cap.isOpened():
        return

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (new_width, new_height))

        results = yolo_model(small_frame, device=device)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Chuyển tọa độ về frame gốc
                x1 = int(x1 * (frame_width / new_width))
                x2 = int(x2 * (frame_width / new_width))
                y1 = int(y1 * (frame_height / new_height))
                y2 = int(y2 * (frame_height / new_height))

                confidence = box.conf[0].cpu().item()
                class_id = int(box.cls[0].cpu().item())

                if class_id == 0 and confidence > 0.5:
                    detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

        tracks = tracker.update_tracks(detections, frame=frame)
        num_people = sum(1 for track in tracks if track.is_confirmed() and track.time_since_update <= 1)

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"People: {num_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("YOLO + DeepSORT Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


detect_and_track_webcam()