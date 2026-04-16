from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()

model = YOLO("app/best.pt") 

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    results = model(image)
    
    return {"detections": results[0].tojson()}