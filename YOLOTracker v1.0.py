import cv2
import json
import time
from collections import deque
from ultralytics import YOLO

class YOLOTracker:
    def __init__(self, model_path, max_json_files=100, lost_target_threshold=100000):
        """
        初始化YOLOTracker类，加载YOLO模型并准备检测。

        :param model_path: YOLO模型路径
        :param max_json_files: 最大保存的JSON文件数量
        :param lost_target_threshold: 目标消失多少帧后，认为目标丢失
        """
        self.model = YOLO(model_path)  # 加载YOLO模型
        self.max_json_files = max_json_files  # 最大保存JSON文件数量
        self.track_frame_ids = {}  # 存储每个 track_id 对应的 frame_id
        self.json_files = deque(maxlen=max_json_files)  # 用于管理JSON文件的队列
        self.lost_targets = {}  # 存储丢失的目标 (track_id -> 失联帧数)
        self.lost_target_threshold = lost_target_threshold  # 目标丢失的最大帧数

    def track_and_generate_json(self, frame, frame_count):
        """
        进行目标检测并返回JSON格式的检测结果。

        :param frame: 当前帧图像
        :param frame_count: 当前帧的计数
        :return: 返回包含检测信息的JSON数据
        """
        results = self.model.track(frame, show=True)  # 使用YOLO进行跟踪检测
        detections = []

        # 记录当前帧出现的目标 track_ids
        active_track_ids = set()

        # 遍历YOLO模型的检测结果
        for result in results:  # results 是一个列表，遍历每个结果
            if result.boxes is not None:  # 确保boxes存在
                boxes = result.boxes.xywh.cpu()  # 获取 xywh 坐标
                track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else []  # 获取track_id
                class_ids = result.boxes.cls.int().cpu().tolist()  # 获取类别的索引（class ids）
                class_names = result.names  # 获取类别名称
                
                # 遍历每个检测框
                for i, box in enumerate(boxes):
                    x, y, w, h = box
                    track_id = track_ids[i] if track_ids else None  # 使用YOLO自带的track_id
                    class_id = class_ids[i]  # 获取当前物体的类索引
                    class_name = class_names[class_id]  # 根据类索引获取物体标签

                    # 如果这是该track_id第一次出现，设置frame_id
                    if track_id not in self.track_frame_ids:
                        self.track_frame_ids[track_id] = frame_count  # 为该track_id分配唯一的frame_id
                    
                    frame_id = self.track_frame_ids[track_id]  # 对于相同的track_id，frame_id不变

                    # 目标重新出现在当前帧时，清除丢失标记
                    if track_id in self.lost_targets:
                        del self.lost_targets[track_id]  # 恢复丢失目标
                        """
                        x：矩形中心的横坐标（x 坐标）
                        y：矩形中心的纵坐标（y 坐标）
                        w：矩形的宽度（width）
                        h：矩形的高度（height）
                        """
                    detection = {
                        "frame_id": frame_id,  # 使用track_id的frame_id
                        "x": x.item(),         # 转换为Python类型
                        "y": y.item(),
                        "w": w.item(),
                        "h": h.item(),
                        "track_id": track_id,  # 保存跟踪ID
                        "class_id": class_id,  # 保存类别ID
                        "class_name": class_name  # 保存类别名称
                    }
                    detections.append(detection)
                    active_track_ids.add(track_id)  # 标记当前帧出现的目标

        # 处理未在当前帧出现的目标（目标消失）
        for track_id in list(self.track_frame_ids.keys()):
            if track_id not in active_track_ids:
                if track_id not in self.lost_targets:
                    self.lost_targets[track_id] = 0  # 初始化丢失帧数
                self.lost_targets[track_id] += 1  # 目标丢失帧数递增

                # 如果目标丢失超过阈值，则删除该目标
                if self.lost_targets[track_id] > self.lost_target_threshold:
                    del self.track_frame_ids[track_id]  # 移除丢失目标的 frame_id
                    del self.lost_targets[track_id]  # 移除丢失目标的记录

        # 构建返回的JSON数据
        json_data = {
            "frame_count": frame_count,  # 帧编号
            "timestamp": time.time(),  # 时间戳
            "detections": detections  # 包含所有检测框的列表
        }

        # 将生成的JSON数据添加到历史记录中
        self.json_files.append((json_data, json_data["timestamp"]))
        return json_data

    def manage_json_history(self):
        """
        管理json历史记录，确保最多保存100个记录，超出时删除最旧的记录。
        """
        # 将deque转换为列表，然后根据时间戳排序
        all_json_files = list(self.json_files)
        all_json_files.sort(key=lambda x: x[1])  # 根据时间戳排序

        # 将排序后的记录重新放回deque
        self.json_files = deque(all_json_files, maxlen=self.max_json_files)

        # 如果文件数量超过最大值，删除最旧的记录
        while len(self.json_files) > self.max_json_files:
            self.json_files.pop()  # 删除最旧的记录


def process_video_with_yolo(camera_id=0 , model_path="yolo11x.pt", max_json_files=100):
    """
    使用YOLO进行摄像头实时视频流处理，持续返回每一帧的检测结果。

    :param camera_id: 摄像头ID，默认是0
    :param model_path: YOLO模型的路径
    :param max_json_files: 最多保存的JSON文件数量
    :return: 一个生成器，持续返回每一帧的检测结果
    """
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    tracker = YOLOTracker(model_path, max_json_files)
    frame_count = 0

    while True:
        # 从摄像头读取一帧
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # 获取检测结果并返回
        json_data = tracker.track_and_generate_json(frame, frame_count)
        tracker.manage_json_history()  # 管理JSON历史记录
        
        frame_count += 1
        
        yield json_data  # 使用yield持续返回JSON数据

    cap.release()

if __name__ == "__main__":
    def run_yolo_detection():
        for frame_data in process_video_with_yolo(camera_id="/dev/video2", model_path="你的模型地址，如果没有请删除这项（将会自动下载）"):
            print(json.dumps(frame_data, indent=4))  # 打印每一帧的检测结果
    run_yolo_detection()