# app/main.py
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from ultralytics import YOLO
import cv2, time, os, io
import numpy as np

# try imageio writer (uses ffmpeg binary from imageio-ffmpeg)
try:
    import imageio
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR.parent / "models" / "yolov8n.pt"
# Where we save processed results (served via /static)
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = STATIC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load YOLO model
try:
    model = YOLO(str(MODEL_PATH))
    print("✅ YOLO model loaded successfully!")
except Exception as e:
    print("❌ Error loading YOLO model:", e)
    model = None

app = FastAPI(title="YOLOv8 Object Detection")

# Mount static so browser can fetch /static/...
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect")
async def detect(request: Request, file: UploadFile = File(...)):
    """
    Single endpoint: accepts image or video (form upload name="file").
    - Images are processed in-memory and returned as a saved jpg in /static/results/.
    - Videos are processed frame-by-frame and saved as MP4 (libx264 via imageio if available).
    """
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")

    content_type = (file.content_type or "").lower()
    file_bytes = await file.read()

    # -------- IMAGE ----------
    if "image" in content_type:
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

        results = model.predict(img, verbose=False)
        annotated = results[0].plot()  # RGB ndarray

        out_name = f"result_{int(time.time())}.jpg"
        out_path = RESULTS_DIR / out_name
        # convert RGB to BGR for OpenCV write
        cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result_image": f"/static/results/{out_name}"}
        )

    # -------- VIDEO ----------
    elif "video" in content_type:
        # save upload temporarily
        tmp_path = str(RESULTS_DIR / f"tmp_upload_{int(time.time())}.mp4")
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            os.remove(tmp_path)
            raise HTTPException(status_code=400, detail="Cannot open uploaded video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        out_name = f"result_{int(time.time())}.mp4"
        out_path = RESULTS_DIR / out_name

        try:
            if HAVE_IMAGEIO:
                # Use imageio (ffmpeg) writer -> more compatible mp4 (H.264)
                writer = imageio.get_writer(
                    str(out_path),
                    fps=float(fps),
                    codec='libx264',
                    ffmpeg_log_level='error',
                    # 'pix_fmt' isn't available as param, but libx264 default is OK;
                    # imageio/ffmpeg will create a playable mp4 with yuv420p.
                )

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # frame is BGR (OpenCV). Convert to RGB for imageio
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # run YOLO on BGR frame (model accepts numpy BGR/RGB; using frame is fine)
                    results = model.predict(frame, verbose=False)
                    annotated = results[0].plot()  # RGB
                    # write RGB frame to writer
                    writer.append_data(annotated)
                writer.close()

            else:
                # fallback: use OpenCV VideoWriter (may not be as widely compatible)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model.predict(frame, verbose=False)
                    annotated = results[0].plot()  # RGB
                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    writer.write(annotated_bgr)
                writer.release()

        except Exception as ex:
            # cleanup and raise
            cap.release()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(str(out_path)):
                try:
                    os.remove(str(out_path))
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail=f"Error writing processed video: {ex}")

        cap.release()
        # remove temporary uploaded file
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        # verify file exists and is non-empty
        if not out_path.exists() or out_path.stat().st_size < 1000:
            raise HTTPException(status_code=500, detail="Processed video is empty or not available.")

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result_video": f"/static/results/{out_name}"}
        )

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload image/video.")
