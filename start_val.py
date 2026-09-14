from ultralytics import YOLO

if __name__ == "__main__":
    pth_path = r"/home/students/master/2024/liuyj/learn/mmcv/ultralytics-yolov11_new/runs/train/DUO+Wtconv/weights/best.pt"

    test_path = r"/home/students/master/2024/liuyj/learn/Dataset/DUO/test_images/scallop/images"
    # Load a model
    model = YOLO(pth_path)  # load a custom model

    # Predict with the model
    results = model(test_path, save=True, conf=0.5, visualize=False)  # predict on an image
