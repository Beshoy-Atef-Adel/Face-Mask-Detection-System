from ultralytics import YOLO

def start_training():
    # 1. تحميل موديل YOLOv8 خفيف (Nano) عشان يخلص بسرعة
    model = YOLO('yolov8n.pt') 

    # 2. بدء التدريب باستخدام ملف الـ yaml اللي عملناه
    # هنقلل الـ epochs لـ 10 بس عشان نلحق نخلص للمشروع
    model.train(
        data='data.yaml', 
        epochs=10, 
        imgsz=640, 
        device='cpu' # لو عندك كارت شاشة Nvidia خليه 0
    )

if __name__ == "__main__":
    start_training()