import cv2
import torch
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


# ============================== Hàm chuyển đổi tọa độ ==============================
def convert_to_custom_coordinates(center_x, center_y, frame_width, frame_height):
    """
    Chuyển đổi tọa độ tâm (center_x, center_y) từ hệ pixel sang hệ tọa độ -1000 đến 1000,
    với tâm khung hình là (0, 0).
    """
    # Dịch chuyển gốc tọa độ về trung tâm
    x_shifted = center_x - (frame_width / 2)
    y_shifted = center_y - (frame_height / 2)

    # Tính giá trị co giãn tối đa
    max_shift_x = frame_width / 2
    max_shift_y = frame_height / 2

    # Co giãn tọa độ về phạm vi -1000 đến 1000
    x_new = (x_shifted / max_shift_x) * 1000
    y_new = (y_shifted / max_shift_y) * 1000

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
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Cập nhật lịch sử vị trí
                track_history[track_id] = track_history.get(track_id, [])[-19:] + [(center_x, center_y)]

                # Tính vector chuyển động
                dx, dy = 0, 0
                if len(track_history[track_id]) >= 2:
                    prev = track_history[track_id][-2]
                    dx = center_x - prev[0]
                    dy = center_y - prev[1]

                # Chuyển đổi tọa độ tâm về hệ -1000 đến 1000
                custom_x, custom_y = convert_to_custom_coordinates(
                    center_x, center_y, FRAME_SIZE[0], FRAME_SIZE[1]
                )

                target_info = {
                    'track_id': track_id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'center_x': center_x,
                    'center_y': center_y,
                    'custom_x': custom_x,
                    'custom_y': custom_y,
                    'frame_width': FRAME_SIZE[0],
                    'frame_height': FRAME_SIZE[1],
                    'bbox_area': area,
                    'movement_vector': (dx, dy)
                }

    return frame, target_info


# ============================== Vòng lặp chính ==============================
track_history = {}
cap = cv2.VideoCapture(VIDEO_PATH)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, target_info = process_frame(frame, track_history)

        # In các giá trị cần thiết để tính toán direction và speed
        if target_info:
            print("\n" + "=" * 60)
            print(f"⚡ THÔNG TIN BÓNG ⚡")
            print(f"- ID bóng: {target_info['track_id']}")
            print(f"- Tọa độ tâm (pixel): ({target_info['center_x']}, {target_info['center_y']})")
            print(f"- Tọa độ tâm (custom): ({target_info['custom_x']:.2f}, {target_info['custom_y']:.2f})")
            print(
                f"- Vector chuyển động: DX = {target_info['movement_vector'][0]: .1f}, DY = {target_info['movement_vector'][1]: .1f}")
            print(f"- Kích thước bóng: {target_info['bbox_area']}px²")
            print("=" * 60)

            # Hiển thị hình ảnh
            # Vẽ bounding box
            cv2.rectangle(processed_frame,
                          (target_info['x1'], target_info['y1']),
                          (target_info['x2'], target_info['y2']),
                          (0, 255, 0), 2)

            # Vẽ tâm và hướng di chuyển
            cv2.circle(processed_frame,
                       (target_info['center_x'], target_info['center_y']),
                       5, (0, 0, 255), -1)

            # # Vẽ vector chuyển động
            dx, dy = target_info['movement_vector']
            # cv2.arrowedLine(processed_frame,
            #                 (target_info['center_x'], target_info['center_y']),
            #                 (int(target_info['center_x'] + dx * 5),
            #                  int(target_info['center_y'] + dy * 5)),
            #                 (255, 0, 0), 2, tipLength=0.3)

            # Vẽ lộ trình
            history = track_history.get(target_info['track_id'], [])
            for i in range(1, len(history)):
                cv2.line(processed_frame, history[i - 1], history[i],
                         (0, 255, 255), 2)

            # Hiển thị thông số
            cv2.putText(processed_frame,
                        f"Center (custom): ({target_info['custom_x']:.2f}, {target_info['custom_y']:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame,
                        f"Movement: DX={dx:.1f}, DY={dy:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame,
                        f"Area: {target_info['bbox_area']}px²", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Robot Controller", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()