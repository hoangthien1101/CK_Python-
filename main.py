import cv2
import torch
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ============================== Cấu hình hệ thống ==============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
YOLO_MODEL_PATH = r"D:\WorkSpace\Homework\NAM3\HK2\LT_Python\Bong.pt"
VIDEO_PATH = r"D:\WorkSpace\Homework\NAM3\HK2\LT_Python\clip_test.mp4"
FRAME_SIZE = (640, 480)
BALL_CLASS_ID = 2
CONF_THRESHOLD = 0.4

# ============================== Khởi tạo mô hình ==============================
yolo_model = YOLO(YOLO_MODEL_PATH).to(DEVICE)
tracker = DeepSort(max_age=10, n_init=3, nn_budget=10)

# ============================== Chuyển đổi tọa độ ==============================
def convert_to_custom_coordinates(center_x, center_y, frame_width, frame_height):
    """ Chuyển tọa độ tâm từ hệ pixel sang hệ -1000 đến 1000 với tâm ảnh là (0, 0). """
    x_new = ((center_x - frame_width / 2) / (frame_width / 2)) * 1000
    y_new = ((center_y - frame_height / 2) / (frame_height / 2)) * 1000
    return x_new, y_new

# ============================== Xử lý frame ==============================
def process_frame(frame, track_history):
    frame = cv2.resize(frame, FRAME_SIZE)

    # Phát hiện vật thể với YOLO
    results = yolo_model(frame, device=DEVICE, half=True, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            if box.cls.item() == BALL_CLASS_ID and box.conf.item() > CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                detections.append(([x1, y1, x2 - x1, y2 - y1], box.conf.item(), BALL_CLASS_ID))

    # Theo dõi với DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)

    target_info = None
    max_area = 0

    for track in tracks:
        if track.is_confirmed():
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            w, h = x2 - x1, y2 - y1
            area = w * h

            if area > max_area:
                max_area = area
                track_id = track.track_id
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Cập nhật lịch sử vị trí
                track_history[track_id] = track_history.get(track_id, [])[-19:] + [(center_x, center_y)]

                # Tính vector chuyển động
                dx, dy = (center_x - track_history[track_id][-2][0], center_y - track_history[track_id][-2][1]) if len(track_history[track_id]) >= 2 else (0, 0)

                # Chuyển đổi tọa độ
                custom_x, custom_y = convert_to_custom_coordinates(center_x, center_y, FRAME_SIZE[0], FRAME_SIZE[1])

                target_info = {
                    'track_id': track_id,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'center_x': center_x, 'center_y': center_y,
                    'custom_x': custom_x, 'custom_y': custom_y,
                    'bbox_area': area, 'movement_vector': (dx, dy)
                }

    return frame, target_info

# ============================== Vòng lặp chính ==============================
track_history = {}
cap = cv2.VideoCapture(VIDEO_PATH)

fps_start_time = time.time()
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, target_info = process_frame(frame, track_history)

        # Tính FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # In thông tin và FPS
        if target_info:
            print("\n" + "=" * 60)
            print(f"⚡ THÔNG TIN BÓNG ⚡")
            print(f"- ID: {target_info['track_id']}")
            print(f"- Tọa độ (pixel): ({target_info['center_x']}, {target_info['center_y']})")
            print(f"- Tọa độ (custom): ({target_info['custom_x']:.2f}, {target_info['custom_y']:.2f})")
            print(f"- Vector chuyển động: DX = {target_info['movement_vector'][0]: .1f}, DY = {target_info['movement_vector'][1]: .1f}")
            print(f"- Kích thước: {target_info['bbox_area']}px²")
            print(f"- 🔥 FPS: {fps:.2f}")
            print("=" * 60)

            # Vẽ bounding box
            cv2.rectangle(processed_frame, (target_info['x1'], target_info['y1']),
                          (target_info['x2'], target_info['y2']), (0, 255, 0), 2)

            # Vẽ tâm và hướng di chuyển
            cv2.circle(processed_frame, (target_info['center_x'], target_info['center_y']), 5, (0, 0, 255), -1)

            # Vẽ lịch sử di chuyển
            history = track_history.get(target_info['track_id'], [])
            for i in range(1, len(history)):
                cv2.line(processed_frame, history[i - 1], history[i], (0, 255, 255), 2)

        # Hiển thị FPS trên màn hình
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Robot Controller", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
