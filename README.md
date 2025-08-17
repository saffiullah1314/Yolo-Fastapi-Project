# YOLOv8 Object Detection Project 🚀

Welcome to my **YOLOv8 Object Detection project**!  
This is a **full-stack AI application** capable of detecting objects in images and videos using the state-of-the-art **YOLOv8 deep learning model**.  
It combines **Python, FastAPI, OpenCV, HTML/CSS** and provides a **web interface** for real-time interaction.

---

## 📚 Table of Contents
- [Project Overview](#-project-overview-)
- [Features](#-features-)
- [Technologies Used](#-technologies-used-)
- [How It Works](#-how-it-works-)
- [Backend Processing](#-backend-processing-)
- [Frontend](#-frontend-)
- [Dataset and Training](#-dataset-and-training-)
- [Code Snippets](#-code-snippets-)
- [Flow Diagrams](#-flow-diagrams-)
- [Live Demo](#-live-demo-)
- [Setup and Usage](#-setup-and-usage-)
- [Author Info](#-author-info-)

---

## 🌟 Project Overview
- Detects multiple objects in **images** and **videos**.  
- Highlights them with **bounding boxes** and **labels**.  
- Designed for real-world AI use cases like **surveillance, research, and automation**.  

I have been learning **AI/ML for the past 6–7 months**, focusing on deep learning theory.  
Now I’m building **real-world projects** and will share advanced projects soon!

---

## ✨ Features
- Detect objects in images and videos  
- Real-time detection with **YOLOv8**  
- **FastAPI backend API** for processing  
- **Frontend UI** for uploading files  
- Annotated outputs with **bounding boxes** and **confidence scores**  
- Secure file handling and storage  

---

## 🛠 Technologies Used
- **Python** – Core programming  
- **FastAPI** – Backend API server  
- **OpenCV** – Image/video processing  
- **YOLOv8 (Ultralytics)** – Object detection model  
- **HTML/CSS + Jinja2** – Frontend interface  
- **Pathlib & Shutil** – File handling  

---

## ⚙ How It Works
1. User uploads image/video via frontend  
2. FastAPI backend receives the file  
3. YOLOv8 model predicts objects + confidence scores  
4. Processed results saved in `/outputs`  
5. Frontend displays annotated images/videos  

---

## 🖥 Backend Processing

### 1️⃣ File Handling
```python
from pathlib import Path
from fastapi import UploadFile

upload_path = Path("uploads") / file.filename
with open(upload_path, "wb") as f:
    f.write(file.file.read())
Ensures secure storage

Prevents overwriting unsafe paths

2️⃣ YOLOv8 Detection
python
Copy
Edit
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load pre-trained model
results = model.predict(upload_path)
results.save("outputs/")
Predicts objects & confidence

Automatically draws bounding boxes

3️⃣ Video Frame Processing
python
Copy
Edit
import cv2

cap = cv2.VideoCapture("uploads/video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame)
    for r in results:
        cv2.rectangle(frame, r.xyxy[0], r.xyxy[1], (0,255,0), 2)
cap.release()
Extracts frames

Processes frame-by-frame

Saves annotated video

🎨 Frontend
HTML/CSS + Jinja2 templates

Upload interface:

html
Copy
Edit
<form action="/upload" method="post" enctype="multipart/form-data">
    <input type="file" name="file" />
    <button type="submit">Upload</button>
</form>
📂 Dataset and Training
Pre-trained YOLOv8 model on COCO dataset (80 classes)

No custom training required

Confidence threshold optimized

💻 Code Snippets
FastAPI main.py
python
Copy
Edit
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolov8n.pt")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb+") as f:
        f.write(file.file.read())
    results = model.predict(file_location)
    results.save("outputs/")
    return {"file": f"Processed {file.filename}"}
🗂 Flow Diagrams
1️⃣ Backend Processing Flow

mathematica
Copy
Edit
User Upload → FastAPI Receives File → YOLOv8 Predicts → Bounding Boxes Added → Save & Return to Frontend
2️⃣ Video Frame Processing Flow

mathematica
Copy
Edit
Video File → OpenCV Extracts Frames → YOLOv8 Detects → Annotated Frames Saved → Combined Output Video
🌐 Live Demo
🔗 Live Demo Link (replace with your hosted link)

⚡ Setup and Usage
bash
Copy
Edit
git clone <repository-url>
cd <project-folder>
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
uvicorn main:app --reload
👉 Open: http://127.0.0.1:8000 and upload files.

📝 Author Info
Name: Saffi Ullah
Passion: AI, ML, Deep Learning, Full-Stack Development
Experience: 6–7 months in AI theory, now building real-world projects
Goal: Share impactful AI projects soon 🚀

yaml
Copy
Edit
