import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ============================================================
# 0. Settings
# ============================================================
DEVICE = "cpu"
MIDAS_MODEL_TYPE = "MiDaS_small"  # MiDaS_small is faster; switch if you need better quality
YOLO_WEIGHTS = "best.pt"

YOLO_CONF_TH = 0.3
YOLO_IOU_TH = 0.45

IMAGE_DIR = Path("images")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Danger threshold: foreground depth / background depth >= DANGER_RATIO
DANGER_RATIO = 1.5

# Depth sampling region inside each bbox (top 0% ~ bottom 70%)
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
def draw_text(img, text, pos, color):
    cv2.putText(
        img, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        5, color, 5, cv2.LINE_AA
    )


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
# 4. Notify helpers (ADB broadcast)
# ============================================================
def send_left():
    """Notify client: go left."""
    cmd = ["adb", "shell", "am", "broadcast", "-a", "com.nav.LEFT"]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print("send_left 失敗:", e)


def send_right():
    """Notify client: go right."""
    cmd = ["adb", "shell", "am", "broadcast", "-a", "com.nav.RIGHT"]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print("send_right 失敗:", e)

def send_either():
    """通知手機：左右皆可"""
    cmd = ["adb", "shell", "am", "broadcast", "-a", "com.nav.EITHER"]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print("send_either 錯誤:", e)
# ============================================================
# 5. Main
# ============================================================
def main():
    imgs = [p for p in IMAGE_DIR.glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]]
    if not imgs:
        print("images/ 沒有圖片")
        return

    for img_path in imgs:
        print(f"\n處理 {img_path.name} ...")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        H, W = bgr.shape[:2]

        # ======================
        # MiDaS depth
        # ======================
        depth = run_midas(bgr)

        # Background depth: 10% percentile
        background_depth = np.percentile(depth, 10)

        # Depth colormap
        dn = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_vis = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

        # ======================
        # YOLO detection
        # ======================
        results = yolo(bgr, conf=YOLO_CONF_TH, iou=YOLO_IOU_TH, verbose=False)
        boxes = results[0].boxes

        out_img = bgr.copy()
        danger_count = 0
        danger_centers = []
        advice_text = "SAFE"
        advice_code = "SAFE"

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            cls = results[0].names[int(box.cls.item())]
            conf = float(box.conf.item())

            d_mean = box_depth_mean(depth, (x1, y1, x2, y2))
            if d_mean is None:
                continue

            ratio = d_mean / background_depth
            is_danger = ratio >= DANGER_RATIO

            if is_danger:
                danger_count += 1
                cx = (x1 + x2) / 2.0
                danger_centers.append((cx, ratio))

            side = box_side((x1, y1, x2, y2), W)
            color = (0, 0, 255) if is_danger else (0, 255, 0)

            cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label = f"{cls} {conf:.2f} d={d_mean:.1f} r={ratio:.2f} {side}"
            if is_danger:
                label = "[D] " + label

            draw_text(out_img, label, (int(x1), max(0, int(y1) - 10)), color)
            draw_text(depth_vis, f"{ratio:.2f}", (int(x1), max(0, int(y1) - 10)), color)

        # ---------- Advice based on weighted danger center ----------
        if danger_count > 0:
            weighted_cx = sum(cx * w for (cx, w) in danger_centers) / (sum(r for (_, r) in danger_centers) + 1e-6)
            center_x = W / 2
            dead_zone = W * 0.12
            offset = weighted_cx - center_x

            if abs(offset) <= dead_zone:
                advice_text = "ADVICE: LEFT OR RIGHT"
                advice_code = "EITHER"
            elif offset > 0:
                advice_text = "ADVICE: GO LEFT"
                advice_code = "LEFT"
            else:
                advice_text = "ADVICE: GO RIGHT"
                advice_code = "RIGHT"

        combined = np.hstack([
            cv2.resize(out_img, (W, H)),
            cv2.resize(depth_vis, (W, H))
        ])

        summary = f"background={background_depth:.1f}  ratio>={DANGER_RATIO}  danger={danger_count}"
        draw_text(combined, summary, (10, 120), (255, 255, 255))
        draw_text(combined, advice_text, (10, 350), (0, 255, 255))

        # ---- ADB broadcast hooks ----
        if advice_code == "LEFT":
            send_left()
        elif advice_code == "RIGHT":
            send_right()
        elif advice_code == "EITHER":
            send_either()
        else:
            # SAFE 之類的狀態，就先不播
            pass

        out_path = OUTPUT_DIR / f"{img_path.stem}_dynratio.png"
        cv2.imwrite(str(out_path), combined)
        print("輸出:", out_path)

        # 清掉已處理的來源檔，避免下次重複跑
        try:
            img_path.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"刪除來源檔失敗 {img_path}: {e}")

    print("\n全部完成")


if __name__ == "__main__":
    main()
