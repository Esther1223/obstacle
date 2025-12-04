from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import io

# ============================================================
# 0. Settings
# ============================================================
DEVICE = "cpu"
MIDAS_MODEL_TYPE = "MiDaS_small"
YOLO_WEIGHTS = "best.pt"

YOLO_CONF_TH = 0.3
YOLO_IOU_TH = 0.45

# Danger threshold
DANGER_RATIO = 1.5

# Depth sampling region inside each bbox
BOX_DEPTH_TOP_RATIO = 0.0
BOX_DEPTH_BOTTOM_RATIO = 0.7

# ============================================================
# 1. Load MiDaS
# ============================================================
print("載入 MiDaS 中...")
midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE)
midas.to(DEVICE)
midas.eval()

midas_tf = torch.hub.load("intel-isl/MiDaS", "transforms")
if MIDAS_MODEL_TYPE in ["DPT_Large", "DPT_Hybrid"]:
    midas_transform = midas_tf.dpt_transform
else:
    midas_transform = midas_tf.small_transform


def run_midas(bgr: np.ndarray) -> np.ndarray:
    """Return MiDaS depth map (H x W)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    batch = midas_transform(rgb).to(DEVICE)

    with torch.no_grad():
        pred = midas(batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

    return pred.squeeze().cpu().numpy().astype(np.float32)


# ============================================================
# 2. Load YOLO
# ============================================================
print("載入 YOLO 中...")
yolo = YOLO(YOLO_WEIGHTS)


# ============================================================
# 3. Utilities
# ============================================================
def box_depth_mean(depth: np.ndarray, bbox):
    """Average depth inside bbox (cropped to 0~70% height)."""
    h, w = depth.shape
    x1, y1, x2, y2 = map(int, bbox)

    box_h = y2 - y1
    if box_h <= 0 or x2 <= x1:
        return None

    sy = int(y1 + box_h * BOX_DEPTH_TOP_RATIO)
    ey = int(y1 + box_h * BOX_DEPTH_BOTTOM_RATIO)

    sy = max(sy, 0)
    ey = min(ey, h)

    patch = depth[sy:ey, x1:x2]
    if patch.size == 0:
        return None
    return float(patch.mean())


def box_side(bbox, W):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    center = W / 2
    dead = W * 0.1
    if cx < center - dead:
        return "left"
    elif cx > center + dead:
        return "right"
    else:
        return "center"


# ============================================================
# 4. Detection Function (核心邏輯)
# ============================================================
def detect_obstacles(bgr: np.ndarray) -> dict:
    """
    處理一張圖片,返回避障建議
    
    Returns:
        {
            "advice": "LEFT" | "RIGHT" | "STRAIGHT" | "EITHER" | "SAFE",
            "danger_count": int,
            "objects": [{"class": str, "confidence": float, "side": str, "is_danger": bool}]
        }
    """
    H, W = bgr.shape[:2]

    # MiDaS depth
    depth = run_midas(bgr)
    background_depth = np.percentile(depth, 10)

    # YOLO detection
    results = yolo(bgr, conf=YOLO_CONF_TH, iou=YOLO_IOU_TH, verbose=False)
    boxes = results[0].boxes

    danger_count = 0
    danger_centers = []
    side_stats = {"left": 0, "center": 0, "right": 0}
    detected_objects = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        cls = results[0].names[int(box.cls.item())]
        conf = float(box.conf.item())

        d_mean = box_depth_mean(depth, (x1, y1, x2, y2))
        if d_mean is None:
            continue

        ratio = d_mean / background_depth
        is_danger = ratio >= DANGER_RATIO
        side = box_side((x1, y1, x2, y2), W)

        if is_danger:
            danger_count += 1
            cx = (x1 + x2) / 2.0
            danger_centers.append((cx, ratio))
            side_stats[side] += 1

        detected_objects.append({
            "class": cls,
            "confidence": round(conf, 2),
            "side": side,
            "is_danger": is_danger,
            "depth_ratio": round(ratio, 2)
        })

    # 決定建議方向
    if danger_count == 0:
        advice_code = "SAFE"
    else:
        left_c = side_stats["left"]
        center_c = side_stats["center"]
        right_c = side_stats["right"]

        # 左右都有障礙物、正中間沒有 → 建議直走
        if left_c > 0 and right_c > 0 and center_c == 0:
            advice_code = "STRAIGHT"
        else:
            # 加權中心邏輯
            xs = [cx for (cx, r) in danger_centers]
            ws = [r for (cx, r) in danger_centers]
            weighted_cx = sum(cx * w for (cx, w) in danger_centers) / (sum(ws) + 1e-6)

            center_x = W / 2.0
            dead_zone = W * 0.12
            offset = weighted_cx - center_x

            if abs(offset) <= dead_zone:
                advice_code = "EITHER"
            elif offset > 0:
                advice_code = "LEFT"
            else:
                advice_code = "RIGHT"

    return {
        "advice": advice_code,
        "danger_count": danger_count,
        "objects": detected_objects
    }


# ============================================================
# 5. FastAPI Server
# ============================================================
app = FastAPI()


@app.get("/")
async def root():
    return {
        "status": "Obstacle Detection Server Running",
        "model": MIDAS_MODEL_TYPE,
        "yolo": YOLO_WEIGHTS
    }


@app.post("/detect")
async def detect_endpoint(file: UploadFile = File(...)):
    """
    接收圖片,返回避障建議
    
    Request: multipart/form-data with 'file' field
    Response: JSON with advice direction
    """
    try:
        # 1. 讀取圖片
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # 2. 轉換成OpenCV格式
        if len(image_np.shape) == 2:  # 灰階圖
            bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.shape[2] == 4:  # RGBA
            bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # 3. 執行偵測
        result = detect_obstacles(bgr)
        
        # 4. 印出結果 (方便debug)
        print(f"🔍 偵測結果: {result['advice']} | 危險物體: {result['danger_count']}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"❌ 錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ============================================================
# 6. 啟動Server
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 障礙物偵測Server啟動中...")
    print("="*50)
    print("📍 本地測試: http://localhost:8000")
    print("📍 API文件: http://localhost:8000/docs")
    print("💡 用 ngrok http 8000 開放外網")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)