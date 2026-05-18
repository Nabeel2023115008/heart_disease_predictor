from ultralytics import YOLO

def detect_heart(image_path):
    model = YOLO("yolov8n.pt")  # Replace with a heart-specific model if available
    results = model(image_path)
    boxes = results[0].boxes.xyxy.cpu().tolist()  # [x1, y1, x2, y2]
    return boxes
