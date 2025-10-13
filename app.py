import io
import traceback
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image

# -------------------------
# Initialize FastAPI application
# -------------------------
app = FastAPI(title="YOLO Ensemble Prediction API")

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load YOLO models
# -------------------------
MODEL_PATHS = [
    "best_3.pt",
    "best_4.pt"
]

models = []
for path in MODEL_PATHS:
    try:
        models.append(YOLO(path))
        print(f"[INFO] Loaded model: {path}")
    except Exception as e:
        print(f"[WARN] Failed to load model {path}: {e}")

# -------------------------
# Ensemble configuration
# -------------------------
IOU_THRESH = 0.5           # IOU threshold for merging boxes
SKIP_BOX_THRESH = 0.5      # Minimum confidence for boxes to be included
WEIGHTS = [1.0 / len(models)] * len(models)
CLASS_NAMES = ['Sweet Basil', 'Basil', 'Holy Basil', 'Unknown']

# -------------------------
# Health check endpoint
# -------------------------
@app.get("/health")
async def health_check():
    """Check server and model status"""
    return {"status": "[INFO] OK", "loaded_models": len(models)}

# -------------------------
# /predict endpoint
# -------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Predict objects in the uploaded image using ensemble of YOLO models.
    Steps:
        1. Read image from request
        2. Run prediction on all loaded models
        3. Normalize and collect boxes, scores, labels
        4. Apply Weighted Boxes Fusion (WBF) for ensemble
        5. Filter boxes below confidence threshold
        6. If multiple classes detected, select the one with highest confidence
    """
    if len(models) == 0:
        return JSONResponse(
            status_code=500,
            content={"error": "[WARN] No models loaded"}
        )

    try:
        print("[START] Reading uploaded image...")
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        print(f"[INFO] Received file: {image.filename}, size: {w}x{h}")

        boxes_list, scores_list, labels_list = [], [], []

        # --------------------------
        # Predict with all models
        # --------------------------
        for idx, model in enumerate(models, start=1):
            results = model.predict(img, conf=0.25, iou=0.45, verbose=False)[0]
            if not hasattr(results, "boxes"):
                print(f"[WARN] Model {idx} returned no boxes")
                continue

            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            labels = results.boxes.cls.cpu().numpy().astype(int)

            # Normalize boxes to 0-1 range
            boxes_norm = boxes.copy()
            boxes_norm[:, [0, 2]] /= w
            boxes_norm[:, [1, 3]] /= h

            boxes_list.append(boxes_norm)
            scores_list.append(scores)
            labels_list.append(labels)

            print(f"[INFO] Model {idx} predicted {len(boxes)} objects")

        # --------------------------
        # Apply Weighted Boxes Fusion (WBF)
        # --------------------------
        boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=WEIGHTS, iou_thr=IOU_THRESH, skip_box_thr=SKIP_BOX_THRESH
        )

        boxes_wbf[:, [0, 2]] *= w
        boxes_wbf[:, [1, 3]] *= h

        # --------------------------
        # Filter by confidence and select class
        # --------------------------
        predictions = []
        for b, s, l in zip(boxes_wbf, scores_wbf, labels_wbf):
            class_name = CLASS_NAMES[int(l)] if int(l) < len(CLASS_NAMES) else "Unknown"
            if s < 0.5:
                continue
            predictions.append({
                "class": class_name,
                "confidence": float(s),
                "box": [float(x) for x in b]
            })

        if len(predictions) == 0:
            predictions.append({"class": "No Object Detected", "confidence": 0.0})

        # If multiple classes detected, select the highest confidence
        if len(predictions) > 1:
            predictions = [max(predictions, key=lambda x: x["confidence"])]

        print(f"[INFO] Ensemble Prediction: {predictions}")
        return JSONResponse(content=predictions)

    except Exception as e:
        detailed_error = traceback.format_exc()
        print("[WARN] ERROR DURING ENSEMBLE PREDICTION")
        print(detailed_error)
        return JSONResponse(
            status_code=500,
            content={
                "error": "[WARN] Prediction failed",
                "details": str(e),
                "trace": detailed_error
            }
        )
