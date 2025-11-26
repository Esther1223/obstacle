import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ============================================================
#  0. 設定
# ============================================================
DEVICE = "cpu"

MIDAS_MODEL_TYPE = "MiDaS_small"   # 可改 MiDaS_small 比較快
YOLO_WEIGHTS = "yolov8n.pt"

YOLO_CONF_TH = 0.3
YOLO_IOU_TH = 0.45

IMAGE_DIR = Path("images")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- 動態比值閾值（你要調的就是這個） ---
DANGER_RATIO = 1.5

# 取 bbox 裡的深度時，不吃地板：只取上方 70% 範圍
BOX_DEPTH_TOP_RATIO = 0.0
BOX_DEPTH_BOTTOM_RATIO = 0.7


# ============================================================
#  1. 載入 MiDaS
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


def run_midas(bgr):
    """回傳 MiDaS 深度圖 (H x W)"""
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
#  2. 載入 YOLO
# ============================================================
print("載入 YOLO 中...")
yolo = YOLO(YOLO_WEIGHTS)


# ============================================================
#  3. 工具
# ============================================================
def draw_text(img, text, pos, color):
    cv2.putText(
        img, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        5, color, 5, cv2.LINE_AA
    )


def box_depth_mean(depth, bbox):
    """取 bbox 上方部分的平均深度"""
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
#  4. 主流程
# ============================================================
def main():
    imgs = [p for p in IMAGE_DIR.glob("*") if p.suffix.lower() in [".jpg",".png",".jpeg",".bmp"]]
    if not imgs:
        print("images/ 沒圖片")
        return

    for img_path in imgs:
        print(f"\n處理 {img_path.name} ...")
        bgr = cv2.imread(str(img_path))
        if bgr is None: continue

        H, W = bgr.shape[:2]

        # ======================
        # MiDaS 深度
        # ======================
        depth = run_midas(bgr)

        # 計算背景深度（取全圖最暗 10% 作為「最遠」）
        background_depth = np.percentile(depth, 10)

        # 深度彩圖
        dn = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_vis = cv2.applyColorMap((dn*255).astype(np.uint8), cv2.COLORMAP_MAGMA)

        # ======================
        # YOLO 偵測
        # ======================
        results = yolo(bgr, conf=YOLO_CONF_TH, iou=YOLO_IOU_TH, verbose=False)
        boxes = results[0].boxes

        out_img = bgr.copy()
        danger_count = 0
        danger_centers = []

        side_stats = {
            "left":   {"count": 0, "max_ratio": 0.0},
            "center": {"count": 0, "max_ratio": 0.0},
            "right":  {"count": 0, "max_ratio": 0.0},
        }


        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            cls = results[0].names[int(box.cls.item())]
            conf = float(box.conf.item())

            # 計算 bbox 深度
            d_mean = box_depth_mean(depth, (x1, y1, x2, y2))
            if d_mean is None:
                continue

            # ======= 動態比值判斷 =======
            ratio = d_mean / background_depth
            is_danger = ratio >= DANGER_RATIO

            if is_danger:
                danger_count += 1
                cx = (x1 + x2) / 2.0
                danger_centers.append((cx, ratio))

            side = box_side((x1, y1, x2, y2), W)
            color = (0,0,255) if is_danger else (0,255,0)

            if is_danger:
                side_stats[side]["count"] += 1
                side_stats[side]["max_ratio"] = max(side_stats[side]["max_ratio"], ratio)

            # 畫框
            cv2.rectangle(out_img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)

            label = f"{cls} {conf:.2f} d={d_mean:.1f} r={ratio:.2f} {side}"
            if is_danger:
                label = "[D] " + label

            draw_text(out_img, label, (int(x1), max(0,int(y1)-10)), color)
            draw_text(depth_vis, f"{ratio:.2f}", (int(x1), max(0,int(y1)-10)), color)
            
             # ---------- 根據危險物的「加權平均中心」給建議 ----------
            if danger_count == 0:
                advice = "SAFE"
            else:
                # 加權平均 x（越近 ratio 越大，權重越大）
                xs = [cx for (cx, r) in danger_centers]
                ws = [r  for (cx, r) in danger_centers]
                weighted_cx = sum(cx * w for cx, w in danger_centers) / (sum(ws) + 1e-6)

                center_x = W / 2.0
                # 這個 dead_zone 是中間「左右都差不多」的範圍
                dead_zone = W * 0.12   # 可微調 0.1 ~ 0.2

                offset = weighted_cx - center_x

                if abs(offset) <= dead_zone:
                    advice = "ADVICE: GO LEFT OR RIGHT"
                elif offset > 0:
                    # 障礙物整體偏右邊 → 往左
                    advice = "ADVICE: GO LEFT"
                else:
                    # 障礙物整體偏左邊 → 往右
                    advice = "ADVICE: GO RIGHT"



        # 合併圖
        combined = np.hstack([
            cv2.resize(out_img, (W, H)),
            cv2.resize(depth_vis, (W, H))
        ])

        summary = f"background={background_depth:.1f}  ratio>={DANGER_RATIO}  danger={danger_count}"
        draw_text(combined, summary, (10,120), (255,255,255))
        draw_text(combined, advice, (10,350), (0,255,255))

        out_path = OUTPUT_DIR / f"{img_path.stem}_dynratio.png"
        cv2.imwrite(str(out_path), combined)
        print("輸出:", out_path)

    print("\n全部完成！")


if __name__ == "__main__":
    main()
