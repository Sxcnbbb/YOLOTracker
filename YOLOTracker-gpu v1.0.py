import cv2
import json
import time
from collections import deque
from ultralytics import YOLO
import torch
import threading
import queue

class YOLOTracker:
    def __init__(self, model_path, max_json_files=100, lost_target_threshold=100000):
        self.model = YOLO(model_path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_json_files = max_json_files
        self.track_frame_ids = {}
        self.json_files = deque(maxlen=max_json_files)
        self.lost_targets = {}
        self.lost_target_threshold = lost_target_threshold

    def track_and_generate_json(self, frame, frame_count):
        # 将帧移动到 GPU
        frame_tensor = torch.from_numpy(frame).to('cuda' if torch.cuda.is_available() else 'cpu')
        frame_tensor = frame_tensor.permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)

        # 使用模型进行跟踪
        results = self.model.track(frame_tensor, show=True)
        detections = []
        active_track_ids = set()

        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else []
                class_ids = result.boxes.cls.int().cpu().tolist()
                class_names = result.names
                
                for i, box in enumerate(boxes):
                    x, y, w, h = box
                    track_id = track_ids[i] if track_ids else None
                    class_id = class_ids[i]
                    class_name = class_names[class_id]

                    if track_id not in self.track_frame_ids:
                        self.track_frame_ids[track_id] = frame_count
                    
                    frame_id = self.track_frame_ids[track_id]

                    if track_id in self.lost_targets:
                        del self.lost_targets[track_id]

                    detection = {
                        "frame_id": frame_id,   
                        "x": x.item(),
                        "y": y.item(),
                        "w": w.item(),
                        "h": h.item(),
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": class_name
                    }
                    detections.append(detection)
                    active_track_ids.add(track_id)

        for track_id in list(self.track_frame_ids.keys()):
            if track_id not in active_track_ids:
                if track_id not in self.lost_targets:
                    self.lost_targets[track_id] = 0
                self.lost_targets[track_id] += 1

                if self.lost_targets[track_id] > self.lost_target_threshold:
                    del self.track_frame_ids[track_id]
                    del self.lost_targets[track_id]

        json_data = {
            "frame_count": frame_count,
            "timestamp": time.time(),
            "detections": detections
        }

        self.json_files.append((json_data, json_data["timestamp"]))
        return json_data


    def manage_json_history(self):
        all_json_files = list(self.json_files)
        all_json_files.sort(key=lambda x: x[1])
        self.json_files = deque(all_json_files, maxlen=self.max_json_files)

        while len(self.json_files) > self.max_json_files:
            self.json_files.pop()

#！！！！！
def video_reader(camera_id, frame_queue):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        frame_queue.put(frame)

    cap.release()

def process_video_with_yolo(camera_id=0, model_path="yolo11x.pt", max_json_files=100):
    frame_queue = queue.Queue()
    tracker = YOLOTracker(model_path, max_json_files)
    frame_count = 0

    # 启动视频读取线程
    reader_thread = threading.Thread(target=video_reader, args=(camera_id, frame_queue))
    reader_thread.start()

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_resized = cv2.resize(frame, (640, 640))

            # 处理帧并生成 JSON 数据
            json_data = tracker.track_and_generate_json(frame_resized, frame_count)
            tracker.manage_json_history()
            frame_count += 1
            
            yield json_data

if __name__ == "__main__":
    #@profile
    def run_yolo_detection():
        for frame_data in process_video_with_yolo(camera_id=0, model_path="输入你自己的模型路径"):
            print(json.dumps(frame_data, indent=4))  # 打印每一帧的检测结果
            
    run_yolo_detection()
