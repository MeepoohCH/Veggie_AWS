import io
import traceback
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image

# =========================
# สร้างแอป FastAPI
# =========================
app = FastAPI(title="YOLO Prediction API")

# ตั้งค่า CORS สำหรับ Next.js หรือ frontend อื่น
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production แนะนำใส่ origin จริง ๆ
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# โหลดโมเดล YOLO
# =========================
try:
    model = YOLO("best.pt")
    print("✅ YOLO Model loaded successfully from best.pt")
except Exception as e:
    print(f"❌ ERROR: Could not load model 'best.pt'. Details: {e}")
    model = None

# =========================
# Health check endpoint
# =========================
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

# =========================
# /predict endpoint
# =========================
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Model Initialization Error",
                "details": "YOLO model failed to load during server startup."
            }
        )

    try:
        # อ่านไฟล์จาก request
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes))
        print(f"📸 Received file: {image.filename}, size: {img.size}")

        # ทำนายด้วย YOLO
        results = model(img)

        # ดึงผลลัพธ์
        predictions = []
        if results and hasattr(results[0], 'boxes'):
            for r in results[0].boxes:
                predictions.append({
                    "class": model.names.get(int(r.cls[0]), "Unknown"),
                    "confidence": float(r.conf[0])
                })

        print(f"✅ Prediction successful. Found {len(predictions)} objects.")
        return JSONResponse(content=predictions)

    except Exception as e:
        detailed_error = traceback.format_exc()
        print("🚨 --- CRITICAL ERROR DURING PREDICTION ---")
        print(detailed_error)
        print("------------------------------------------")

        return JSONResponse(
            status_code=500,
            content={
                "error": "Prediction processing failed in FastAPI",
                "details": str(e),
                "trace": detailed_error
            }
        )


