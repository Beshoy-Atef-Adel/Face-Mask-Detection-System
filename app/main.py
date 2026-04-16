from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI(title="Face Mask Detection API")

model = YOLO("app/best.pt")


@app.get("/")
async def root():
    return {"message": "Face Mask Detection API is running. Use POST /predict to detect masks."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "name": result.names[cls_id],
            "class": cls_id,
            "confidence": float(box.conf[0]),
            "box": {
                "x1": float(box.xyxy[0][0]),
                "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]),
                "y2": float(box.xyxy[0][3]),
            },
        })

    return {"detections": detections, "count": len(detections)}
