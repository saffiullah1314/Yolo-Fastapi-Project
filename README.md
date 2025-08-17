YOLOv8 Object Detection Project 🚀

Welcome to my YOLOv8 Object Detection project! This is a full-stack AI application capable of detecting objects in images and videos using state-of-the-art YOLOv8 deep learning model. It combines Python, FastAPI, OpenCV, HTML/CSS, and provides a web interface for real-time interaction.


---

Table of Contents 📚

Project Overview

Features

Technologies Used

How It Works

Backend Processing

Frontend

Dataset and Training

Code Snippets

Flow Diagrams

Live Demo

Setup and Usage

Author Info



---

Project Overview 🌟

This project detects multiple objects in images or videos and highlights them with bounding boxes and labels.
It is designed for practical AI applications such as surveillance, research, and automation.

I have been working on AI/ML for the past 6–7 months, focusing on theory and understanding deep learning concepts. Now I am building real-world projects and plan to share advanced, heavy projects soon.


---

Features ✨

Detect objects in images and videos.

Real-time detection with YOLOv8.

Backend API using FastAPI for processing.

Frontend interface for uploading files.

Annotated output images/videos with bounding boxes and confidence scores.

Secure file handling and storage.



---

Technologies Used 🛠

Python – Core programming for AI & backend.

FastAPI – Backend API server.

OpenCV – Image/video processing.

YOLOv8 (Ultralytics) – Object detection model.

HTML/CSS + Jinja2 – Frontend interface.

Pathlib & Shutil – File handling in backend.



---

How It Works ⚙

1. User uploads image/video via frontend.


2. FastAPI backend receives the file.


3. YOLOv8 model predicts objects and confidence scores.


4. Processed results are saved and returned to frontend.


5. Frontend displays annotated images/videos.




---

Backend Processing 🖥

1️⃣ File Handling

from pathlib import Path
from fastapi import UploadFile

upload_path = Path("uploads") / file.filename
with open(upload_path, "wb") as f:
    f.write(file.file.read())

Ensures secure file storage.

Prevents overwriting or unsafe paths.


2️⃣ YOLOv8 Detection

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load pre-trained model
results = model.predict(upload_path)
results.save("outputs/")

Model predicts objects and confidence scores.

Bounding boxes drawn automatically.

Video frames are processed one by one.


3️⃣ Video Frame Processing

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

Extracts frames from video.

Processes frame by frame.

Saves annotated video at the end.



---

Frontend 🎨

HTML/CSS + Jinja2 templates for uploading files.

Users can see processed images/videos without page reloads.


<form action="/upload" method="post" enctype="multipart/form-data">
    <input type="file" name="file" />
    <button type="submit">Upload</button>
</form>


---

Dataset and Training 📂

Uses pre-trained YOLOv8 model on COCO dataset (80 classes).

No custom training required for this project.

Confidence threshold set to optimize detection accuracy.



---

Code Snippets 💻

FastAPI main.py sample:

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


---

Flow Diagrams 🗂

1️⃣ Backend Processing Flow

User Upload
     |
     v
FastAPI Receives File
     |
     v
YOLOv8 Predicts Objects
     |
     v
Bounding Boxes Added
     |
     v
Save & Send Result to Frontend

2️⃣ Video Frame Processing Flow

Video File
     |
     v
OpenCV Extracts Frames
     |
     v
YOLOv8 Detects Objects Frame-by-Frame
     |
     v
Annotated Frames Saved
     |
     v
Combine Frames -> Output Video


---

Live Demo 🌐

Try the project live here:
[Live Demo Link] – replace with your hosted link


---

Setup and Usage ⚡

git clone <repository-url>
cd <project-folder>
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
uvicorn main:app --reload

Open http://127.0.0.1:8000 and upload images/videos.


---

Author Info 📝

Name: Saffi Ullah
Passion: AI, Machine Learning, Deep Learning, Full-Stack Development
Experience: 6–7 months learning AI theory, now building real-world projects
Goal: Share heavy, impactful AI projects soon
