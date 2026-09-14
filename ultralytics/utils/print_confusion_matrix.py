import cv2
import supervision as sv
import numpy as np
from inference import get_model  # 假设从该模块获取模型，需根据实际情况调整

# 步骤2：加载数据集（替换为实际数据集路径）
dataset = sv.detection_dataset.from_yolo(
    images_path="your_dataset_images_path",
    annotations_path="your_dataset_annotations_path"
)

# 步骤3：获取YOLO11模型（替换为实际模型ID或加载方式）
model = get_model(model_id="yolo11s - 640")  # 若通过Roboflow获取，替换为对应模型ID

def callback(image: np.ndarray) -> sv.Detections:
    result = model.infer(image)[0]  # 模型推理
    detections = sv.Detections.from_inference(result)  # 转换为supervision的检测格式
    return detections

# 步骤4：生成并绘制混淆矩阵
confusion_matrix = sv.confusion_matrix.benchmark(
    dataset=dataset,
    callback=callback
)
confusion_matrix.plot()