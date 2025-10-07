from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import traceback

app = Flask(__name__)

# โหลดโมเดล
try:
    model = YOLO("best.pt")
    print("YOLO Model loaded successfully from best.pt")
except Exception as e:
    print(f"ERROR: Could not load model 'best.pt'. Please check the file path. Details: {e}")
    model = None

@app.route("/predict", methods=["POST"])
def predict():
    # 1. ตรวจสอบสถานะโมเดล
    if model is None:
        return jsonify({
            "error": "Model Initialization Error",
            "details": "YOLO model failed to load during server startup."
        }), 500

    try:
        # 2. ตรวจสอบและรับไฟล์ ใช้ request.files["image"] โดยตรงตามที่ Next.js ส่งมา
        if "image" not in request.files:
            print("ERROR: File key 'image' not found in request.files.")
            return jsonify({ 
                "error": "Missing 'image' file in request.files",
                "details": "The Flask server did not receive the file with the expected key 'image'."
            }), 400
            
        file = request.files["image"]
        
        # 3. อ่านและเปิดไฟล์ด้วย PIL ใช้ file.read() เพื่ออ่านไบนารี และใช้ io.BytesIO เพื่อให้ PIL เปิดได้
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        print(f"Received file: {file.filename}, Image size: {img.size}")

        # 4. การประมวลผลโมเดล
        results = model(img)

        # 5. การดึงผลลัพธ์
        predictions = []
        if results and hasattr(results[0], 'boxes'):
            for r in results[0].boxes:
                predictions.append({
                    "class": model.names.get(int(r.cls[0]), "Unknown"),
                    "confidence": float(r.conf[0])
                })
        
        print(f"Prediction successful. Found {len(predictions)} objects.")
        return jsonify(predictions)

    except Exception as e:
        detailed_error = traceback.format_exc()
        print(f"--- CRITICAL ERROR DURING PREDICTION ---")
        print(detailed_error)
        print("------------------------------------------")
        
        # ส่ง 500 Internal Server Error พร้อมรายละเอียดกลับไป
        return jsonify({
            "error": "Prediction processing failed in Flask",
            "details": str(e),
            "trace": detailed_error
        }), 500


if __name__ == "__main__":
    # สำคัญ: เพื่อให้สามารถดีบักได้ง่าย ควรตั้งค่า CORS ในสภาพแวดล้อมจริง
    # สำหรับการทดสอบ localhost จะไม่ต้องตั้งค่า CORS แต่ถ้ายังไม่ได้ผล ให้ลองใช้ไลบรารี flask_cors
    app.run(host="0.0.0.0", port=5000)
