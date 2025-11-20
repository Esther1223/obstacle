import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import json
import argparse

# ========== 初始化模型 ==========
CALIBRATION_VERSION = 3  # bump when depth processing changes
# Detection thresholds
CONF_THRESH = 0.25
IOU_THRESH = 0.5

def initialize_models():
    print("=" * 60)
    print("🚦 障礙物偵測系統 - 距離校正版")
    print("=" * 60)
    print("\n⏳ 正在載入模型...")
    
    yolo_model = YOLO('yolov8n.pt')
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    
    print("✅ 模型載入完成")
    print(f"🖥️  使用裝置: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
    
    return yolo_model, midas, transform, device

# ========== Robust ROI depth helper ==========
def robust_roi_depth(depth_map, x1, y1, x2, y2, inner_ratio=0.8):
    """Compute a robust depth statistic inside a shrunken ROI.

    - Shrinks the bbox by inner_ratio to reduce edge/background contamination.
    - Filters non-finite and non-positive values.
    - Uses 20-80 percentile trimmed median for robustness.
    Returns float depth or None if insufficient data.
    """
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    # shrink box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int((1.0 - inner_ratio) * 0.5 * bw)
    pad_y = int((1.0 - inner_ratio) * 0.5 * bh)
    sx1 = max(0, x1 + pad_x)
    sy1 = max(0, y1 + pad_y)
    sx2 = min(w - 1, x2 - pad_x)
    sy2 = min(h - 1, y2 - pad_y)
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    region = depth_map[sy1:sy2, sx1:sx2].astype(np.float32)
    vals = region[np.isfinite(region) & (region > 0)]
    if vals.size == 0:
        return None
    q1, q3 = np.percentile(vals, [20, 80])
    trimmed = vals[(vals >= q1) & (vals <= q3)]
    if trimmed.size == 0:
        trimmed = vals
    return float(np.median(trimmed))

# ========== 計算深度值 ==========
def calculate_depth(image_path, midas, transform, device, yolo_model):
    """計算照片中所有物體的平均深度值"""
    frame = cv2.imread(image_path)
    if frame is None:
        return None, None, None
    
    # 調整大小
    frame = cv2.resize(frame, (640, 480))
    
    # YOLOv8 偵測
    # Standardized inference frame
    proc_w, proc_h = 640, 480
    proc_frame = cv2.resize(frame, (proc_w, proc_h))
    results = yolo_model(proc_frame, verbose=False, conf=CONF_THRESH, iou=IOU_THRESH)
    detections = results[0].boxes
    
    # MiDaS 深度估計
    img_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy().astype(np.float32)
    
    # 找到最大的物體
    max_area = 0
    target_depth = None
    selected_center_x = None
    
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        
        if confidence < CONF_THRESH:
            continue
        
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            robust_depth = robust_roi_depth(depth_map, x1, y1, x2, y2)
            if robust_depth is not None:
                target_depth = robust_depth
                selected_center_x = (x1 + x2) / 2.0
    
    xn = None
    if selected_center_x is not None:
        frame_center_x = frame.shape[1] / 2.0
        if frame_center_x > 0:
            xn = (selected_center_x - frame_center_x) / frame_center_x
        else:
            xn = 0.0
    
    return target_depth, frame, xn

# ========== 校正模式 ==========
def calibration_mode(yolo_model, midas, transform, device):
    """執行距離校正"""
    print("=" * 60)
    print("📏 校正模式")
    print("=" * 60)
    print("\n說明：")
    print("  1. 請準備一個物體（例如：椅子、箱子）")
    print("  2. 將物體放在 3 個不同的已知距離")
    print("  3. 每個距離拍一張照片")
    print("  4. 系統會自動計算校正公式\n")
    print("建議距離：1公尺、2公尺、3公尺")
    print("-" * 60)
    
    input("\n按 Enter 開始校正...")
    
    calibration_data = []
    existing_config = load_calibration()
    if existing_config and existing_config.get('calibration_data'):
        existing_entries = existing_config.get('calibration_data', [])
        existing_count = len(existing_entries)
        while True:
            choice = input(f"\n偵測到既有校正資料，共 {existing_count} 筆，是否要在此基礎上新增？(Y/n): ").strip().lower()
            if choice in ("", "y", "yes"):
                calibration_data = [dict(entry) for entry in existing_entries]
                print(f"➡️  將沿用既有資料 {existing_count} 筆，並再加入 15 筆新的校正樣本。")
                break
            elif choice in ("n", "no"):
                calibration_data = []
                print("↩️  將清除既有資料並重新開始校正。")
                break
            else:
                print("請輸入 Y 或 N。")
    else:
        print("\n目前沒有已儲存的校正資料，將重新開始。")
    
    for i in range(15):
        print(f"\n{'=' * 60}")
        print(f"第 {i+1}/15 張校正照片")
        print("=" * 60)
        
        # 輸入實際距離
        while True:
            try:
                actual_distance = float(input(f"請輸入物體的實際距離（公尺）: "))
                if actual_distance > 0:
                    break
                else:
                    print("❌ 距離必須大於 0")
            except ValueError:
                print("❌ 請輸入有效的數字")
        
        # 輸入照片路徑
        while True:
            image_path = input("請輸入照片路徑（或拖曳照片到此）: ").strip().strip('"').strip("'")
            
            if os.path.exists(image_path):
                depth_value, frame, xn = calculate_depth(image_path, midas, transform, device, yolo_model)
                
                if depth_value is not None:
                    calibration_data.append({
                        'distance': float(actual_distance),
                        'depth': float(depth_value),
                        'xn': float(xn) if xn is not None else 0.0
                    })
                    print(f"✅ 成功！深度值: {depth_value:.4f}")
                    
                    # 顯示照片預覽
                    cv2.imshow(f'Calibration {i+1}', frame)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()
                    break
                else:
                    print("❌ 無法偵測到物體，請重新拍攝")
            else:
                print("❌ 找不到檔案")
    
    # ========== 計算校正公式 ==========
    print("\n" + "=" * 60)
    print("🔧 正在計算校正公式...")
    print("=" * 60)
    
    depths = np.array([d['depth'] for d in calibration_data])
    distances = np.array([d['distance'] for d in calibration_data])
    
    inv_depths = 1.0 / depths
    A = np.vstack([inv_depths, np.ones(len(inv_depths))]).T
    params, residuals, rank, s = np.linalg.lstsq(A, distances, rcond=None)

    # One-pass robust refit based on residuals (trim outliers)
    pred = params[0] * inv_depths + params[1]
    err = np.abs(pred - distances)
    med = np.median(err) if err.size else 0.0
    thresh = 2.5 * max(med, 1e-6)
    keep = err <= thresh
    if keep.sum() >= 3 and keep.sum() < len(inv_depths):
        A2 = A[keep]
        d2 = distances[keep]
        params, _, _, _ = np.linalg.lstsq(A2, d2, rcond=None)

    a, b = float(params[0]), float(params[1])

    # Fallback: if intercept leads to non-physical predictions on calibration data,
    # refit without intercept (b = 0) to enforce monotonic relation.
    pred_check = a * inv_depths + b
    if (pred_check <= 0).any() or a <= 0:
        a_numer = float(np.sum(distances * inv_depths))
        a_denom = float(np.sum(inv_depths ** 2) + 1e-9)
        a = a_numer / a_denom
        b = 0.0

    # Edge bias correction using lateral position (horizontal only, allow asymmetry)
    xns = np.array([float(d.get('xn', 0.0)) for d in calibration_data])
    d_pred = a / depths + b
    X = d_pred * xns
    Y = d_pred * (xns ** 2)
    valid = np.abs(d_pred) > 1e-6
    if np.sum(valid) >= 2 and (np.abs(X[valid]).sum() > 1e-6 or np.abs(Y[valid]).sum() > 1e-6):
        M = np.vstack([X[valid], Y[valid]]).T
        target = distances[valid] - d_pred[valid]
        corr_params, _, _, _ = np.linalg.lstsq(M, target, rcond=None)
        k1 = float(corr_params[0]) if len(corr_params) > 0 else 0.0
        k2 = float(corr_params[1]) if len(corr_params) > 1 else 0.0
    else:
        k1 = 0.0
        k2 = 0.0
    k1 = float(np.clip(k1, -0.5, 0.5))
    k2 = float(np.clip(k2, -0.5, 0.5))
    
    print("\n校正結果：")
    print("-" * 60)
    for i, data in enumerate(calibration_data):
        predicted = a / data['depth'] + b
        xn_i = float(data.get('xn', 0.0))
        predicted_corr = predicted * (1.0 + k1 * xn_i + k2 * (xn_i ** 2))
        error = abs(predicted_corr - data['distance'])
        print(f"  {i+1}. 實: {data['distance']:.2f}m | 深度: {data['depth']:.4f} | 預測(水平校正): {predicted_corr:.2f}m | 誤差: {error:.2f}m")

    avg_error = np.mean([
        abs(((a / d['depth'] + b) * (1.0 + k1 * float(d.get('xn', 0.0)) + k2 * (float(d.get('xn', 0.0)) ** 2))) - d['distance'])
        for d in calibration_data
    ])
    print(f"\n平均誤差 (水平校正後): {avg_error:.2f}m")

    
    # 儲存校正資料
    calibration_config = {
        'version': CALIBRATION_VERSION,
        'a': a,
        'b': b,
        'k1': k1,
        'k2': k2,
        'calibration_data': calibration_data
    }
    
    try:
        with open('calibration.json', 'w', encoding='utf-8') as f:
            json.dump(calibration_config, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 校正完成！公式已儲存至 calibration.json")
        print(f"   公式: distance = {a:.4f} / depth + ({b:.4f})")
    except Exception as e:
        print(f"\n⚠️  儲存失敗: {e}")
        print("但可以繼續使用（只是下次需要重新校正）")
    
    return calibration_config

# ========== 載入校正資料 ==========
def load_calibration():
    """載入校正設定"""
    if os.path.exists('calibration.json'):
        try:
            with open('calibration.json', 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    print("⚠️  校正檔案是空的")
                    os.remove('calibration.json')
                    return None
                data = json.loads(content)
                # Invalidate old calibration if version mismatch
                if not isinstance(data, dict) or data.get('version') != CALIBRATION_VERSION:
                    print("�`��  �ե�����檔�榡/���ܦX�P�ɮסA�ݨ�^�s�ե�.")
                    try:
                        os.remove('calibration.json')
                    except:
                        pass
                    return None
                return data
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  校正檔案損壞: {e}")
            print("將刪除舊檔案並重新校正...")
            try:
                os.remove('calibration.json')
            except:
                pass
            return None
        except Exception as e:
            print(f"⚠️  載入失敗: {e}")
            return None
    return None

# ========== 偵測模式 ==========
def detection_mode(yolo_model, midas, transform, device, calibration_config, image_path=None, show_window=True, output_dir=None):
    """使用校正後的公式進行偵測"""
    print("\n" + "=" * 60)
    print("🎯 偵測模式")
    print("=" * 60)
    print(f"使用校正公式: distance = {calibration_config['a']:.4f} / depth + ({calibration_config['b']:.4f})\n")
    
    WARNING_DISTANCE = 1.5
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    
    # 輸入照片
    provided_path = bool(image_path)
    while True:
        candidate_path = image_path if image_path else input("請輸入要偵測的照片路徑: ").strip().strip('"').strip("'")
        
        if candidate_path and os.path.exists(candidate_path):
            image_path = candidate_path
            break
        else:
            if provided_path:
                print("❌ 找不到檔案")
                return None
            print("❌ 找不到檔案")
    
    # 讀取照片
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ 無法讀取照片")
        return
    
    print(f"✅ 成功讀取: {os.path.basename(image_path)}")
    
    # 調整大小
    # Standardized inference frame (keep fixed for YOLO/MiDaS)
    proc_w, proc_h = 640, 480
    proc_frame = cv2.resize(frame, (proc_w, proc_h))
    original_height, original_width = frame.shape[:2]
    aspect_ratio = original_width / original_height
    
    if aspect_ratio > WINDOW_WIDTH / WINDOW_HEIGHT:
        new_width = WINDOW_WIDTH
        new_height = int(WINDOW_WIDTH / aspect_ratio)
    else:
        new_height = WINDOW_HEIGHT
        new_width = int(WINDOW_HEIGHT * aspect_ratio)
    
    frame = cv2.resize(frame, (new_width, new_height))
    
    # YOLOv8 偵測
    print("⏳ 正在偵測障礙物...")
    results = yolo_model(proc_frame, verbose=False, conf=CONF_THRESH, iou=IOU_THRESH)
    detections = results[0].boxes
    # Fallback: if nothing detected, try lower conf
    try:
        if detections is None or len(detections) == 0:
            results = yolo_model(proc_frame, verbose=False, conf=0.10, iou=IOU_THRESH)
            detections = results[0].boxes
            if detections is not None and len(detections) > 0:
                print("? tUXv (conf=0.10)")
    except Exception:
        pass
    
    # MiDaS 深度估計
    print("⏳ 正在估算深度...")
    img_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy().astype(np.float32)
    
    # 處理偵測結果
    print("⏳ 正在分析...")
    print("\n" + "=" * 60)
    print("偵測結果")
    print("=" * 60)
    
    frame_center_x = proc_frame.shape[1] / 2
    danger_objects = []
    detection_count = 0
    
    a = calibration_config['a']
    b = calibration_config['b']
    k1 = float(calibration_config.get('k1', 0.0))
    k2 = float(calibration_config.get('k2', 0.0))
    
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        
        if confidence < CONF_THRESH:
            continue
        
        detection_count += 1
        center_x = (x1 + x2) / 2
        
        # 計算深度
        avg_depth = robust_roi_depth(depth_map, x1, y1, x2, y2)
        if avg_depth is None or avg_depth <= 0:
            continue
        
        # 使用校正公式計算距離
        if avg_depth > 0:
            distance = a / avg_depth + b
            # Lateral bias correction (horizontal angle only)
            xn = (center_x - frame_center_x) / frame_center_x if frame_center_x > 0 else 0.0
            distance = distance * (1.0 + k1 * xn + k2 * (xn ** 2))
            distance = max(0.1, min(distance, 50.0))
        else:
            distance = 999
        
        # 判斷顏色
        if distance < WARNING_DISTANCE:
            color = (0, 0, 255)
            status = "Danger"
        elif distance < 3.0:
            color = (0, 165, 255)
            status = "Caution"
        else:
            color = (0, 255, 0)
            status = "Safe"
        
        # 繪製框框
        cv2.rectangle(proc_frame, (x1, y1), (x2, y2), color, 4)
        
        # 顯示距離
        label = f"{distance:.1f}m"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.rectangle(proc_frame, (x1, y1 - label_size[1] - 10), 
                      (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(proc_frame, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # 終端機輸出
        print(f"\n{detection_count}. [{status}]")
        print(f"   距離: {distance:.1f}m")
        print(f"   深度值: {avg_depth:.4f}")
        
        if distance < WARNING_DISTANCE:
            danger_objects.append({'distance': distance, 'center_x': center_x})
            print(f"   ⚠️ 警告: 距離過近！")
    
    print("\n" + "=" * 60)
    print(f"✅ 偵測到 {detection_count} 個障礙物")
    
    # 判斷躲避方向
    # Prepare display frame from processing frame
    frame = cv2.resize(proc_frame, (new_width, new_height))
    if danger_objects:
        print(f"⚠️ 有 {len(danger_objects)} 個危險障礙物")
        
        left_danger = [obj for obj in danger_objects if obj['center_x'] < frame_center_x]
        right_danger = [obj for obj in danger_objects if obj['center_x'] >= frame_center_x]
        
        if len(left_danger) < len(right_danger):
            escape_direction = "LEFT"
            arrow_start = (new_width - 100, new_height // 2)
            arrow_end = (100, new_height // 2)
        elif len(right_danger) < len(left_danger):
            escape_direction = "RIGHT"
            arrow_start = (100, new_height // 2)
            arrow_end = (new_width - 100, new_height // 2)
        else:
            left_min = min([o['distance'] for o in left_danger]) if left_danger else 999
            right_min = min([o['distance'] for o in right_danger]) if right_danger else 999
            if left_min > right_min:
                escape_direction = "LEFT"
                arrow_start = (new_width - 100, new_height // 2)
                arrow_end = (100, new_height // 2)
            else:
                escape_direction = "RIGHT"
                arrow_start = (100, new_height // 2)
                arrow_end = (new_width - 100, new_height // 2)
        
        print(f"🚨 建議: 往{'左' if escape_direction == 'LEFT' else '右'}閃避")
        
        # 繪製超大箭頭
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, new_height // 2 - 80), 
                      (new_width, new_height // 2 + 80), (0, 100, 200), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        cv2.arrowedLine(frame, arrow_start, arrow_end, 
                       (0, 255, 255), 25, tipLength=0.15)
        
        text = f"GO {escape_direction}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5)[0]
        text_x = (new_width - text_size[0]) // 2
        text_y = new_height // 2 - 100
        
        cv2.putText(frame, text, (text_x + 3, text_y + 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5)
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5)
        
        cv2.rectangle(frame, (0, 0), (new_width, 70), (0, 0, 200), -1)
        cv2.putText(frame, "! DANGER - OBSTACLE DETECTED !", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    else:
        print("✅ 安全")
        cv2.rectangle(frame, (0, 0), (new_width, 70), (0, 200, 0), -1)
        cv2.putText(frame, "SAFE - No Danger", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # 顯示結果
    if show_window:
        cv2.namedWindow('Obstacle Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Obstacle Detection', new_width, new_height)
        cv2.imshow('Obstacle Detection', frame)
    
    # 儲存結果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_{base_name}_calibrated.jpg")
    else:
        output_path = f"result_{base_name}_calibrated.jpg"
    cv2.imwrite(output_path, frame)
    print(f"💾 已儲存: {output_path}")
    
    if show_window:
        print("\n按任意鍵關閉...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
    return output_path

# ========== 批次處理 ==========
def batch_process_folder(input_dir, output_dir, yolo_model, midas, transform, device, calibration_config, show_window=False):
    """Process every image in input_dir and save annotated results to output_dir."""
    print("\n" + "=" * 60)
    print("📂 批次偵測模式")
    print("=" * 60)
    print(f"來源資料夾: {input_dir}")
    print(f"輸出資料夾: {output_dir}")
    
    if not os.path.isdir(input_dir):
        print("❌ 找不到來源資料夾")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(supported_ext)
    ]
    
    if not image_files:
        print("ℹ️ 資料夾中沒有影像檔")
        return
    
    image_files.sort()
    
    for idx, img_path in enumerate(image_files, start=1):
        print(f"\n[{idx}/{len(image_files)}] 處理 {os.path.basename(img_path)}")
        detection_mode(
            yolo_model,
            midas,
            transform,
            device,
            calibration_config,
            image_path=img_path,
            show_window=show_window,
            output_dir=output_dir,
        )

# ========== 主程式 ==========
def main():
    parser = argparse.ArgumentParser(description="Obstacle detection / calibration tool")
    parser.add_argument("--batch", "-b", type=str, help="指定資料夾後批次處理其中的影像")
    parser.add_argument("--output", "-o", type=str, default="result", help="輸出標註影像的資料夾")
    parser.add_argument("--nogui", action="store_true", help="批次模式下不顯示 OpenCV 視窗")
    args = parser.parse_args()
    
    yolo_model, midas, transform, device = initialize_models()
    
    calibration_config = load_calibration()
    
    # 批次模式：有指定 batch 參數時直接處理資料夾並結束
    if args.batch:
        if not calibration_config:
            print("❌ 未找到校正資料，需要先進行校正")
            return
        batch_process_folder(
            args.batch,
            args.output,
            yolo_model,
            midas,
            transform,
            device,
            calibration_config,
            show_window=not args.nogui,
        )
        return
    
    if calibration_config:
        print("✅ 找到校正資料")
        print(f"   公式: distance = {calibration_config['a']:.4f} / depth + ({calibration_config['b']:.4f})")
        print("\n請選擇模式：")
        print("  1. 重新校正")
        print("  2. 使用現有校正進行偵測")
        choice = input("\n輸入選項 (1/2): ").strip()
        
        if choice == "1":
            calibration_config = calibration_mode(yolo_model, midas, transform, device)
    else:
        print("❌ 未找到校正資料，需要先進行校正")
        calibration_config = calibration_mode(yolo_model, midas, transform, device)
    
    # 進入偵測模式
    while True:
        detection_mode(yolo_model, midas, transform, device, calibration_config)
        
        again = input("\n要繼續偵測其他照片嗎? (y/n): ").lower()
        if again != 'y':
            break
    
    print("\n✅ 程式結束")

if __name__ == "__main__":
    main()










