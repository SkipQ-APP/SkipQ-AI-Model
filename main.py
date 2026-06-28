from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your local frontend file to access the API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load YOLOv8 model (downloads automatically on first run)
model = YOLO("yolov10x.pt") 

@app.get("/")
def read_root():
    return {"message": "AI People Counter API is running!"}

@app.post("/count-people")
async def count_people(file: UploadFile = File(...)):
    # 1. Read the uploaded image bytes
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    results = model(image)
    
    detected_classes = results[0].boxes.cls.int().tolist()
    person_count = detected_classes.count(0)
    
    # 4. Return JSON response to the Node.js backend
    return {
        "status": "success",
        "people_count": person_count
    }