import cv2
import numpy as np
import os
import glob
import json
from scipy.optimize import least_squares

def load_and_crop(image_path, crop_fraction=0.25):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = img.shape[:2]
    x_start = int(width * crop_fraction)
    x_end = int(width * (1 - crop_fraction))
    y_start = int(height * crop_fraction)
    y_end = int(height * (1 - crop_fraction))
    return img[y_start:y_end, x_start:x_end]

def threshold_blue_or_dark(img_bgr, hue_min=90, hue_max=140, sat_min=30, val_min=50, dark_threshold=80):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue_low = np.array([hue_min, sat_min, val_min], dtype=np.uint8)
    blue_high = np.array([hue_max, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, blue_low, blue_high)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_or(blue_mask, dark_mask)
    return combined

def clean_mask(mask, closing_kernel=(5,5), dilate_iter=1, erode_iter=1):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, closing_kernel)
    m = mask.copy()
    if dilate_iter:
        m = cv2.dilate(m, k, iterations=dilate_iter)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if erode_iter:
        m = cv2.erode(m, k, iterations=erode_iter)
    return m

def compute_row_midpoints(mask, row_step=1):
    h, w = mask.shape
    pts = []
    for y in range(0, h, row_step):
        x_inds = np.where(mask[y, :] > 0)[0]
        if len(x_inds) >= 2:
            left = x_inds[0]
            right = x_inds[-1]
            center_x = (left + right) / 2.0
            pts.append((center_x, y))
    return np.array(pts, dtype=float)

def fit_cubic_bezier(points):
    P0 = points[0]
    P3 = points[-1]

    def bezier_curve(t_vals, P0, P1, P2, P3):
        u = 1 - t_vals
        return (u**3)[:, None] * P0 + \
               3 * (u**2 * t_vals)[:, None] * P1 + \
               3 * (u * t_vals**2)[:, None] * P2 + \
               (t_vals**3)[:, None] * P3

    def loss(control_vars, observed_pts):
        P1 = control_vars[0:2]
        P2 = control_vars[2:4]
        ts = np.linspace(0, 1, len(observed_pts))
        curve = bezier_curve(ts, P0, P1, P2, P3)
        return (curve - observed_pts).ravel()

    guess = np.concatenate([P0 + (P3 - P0) / 3, P0 + 2*(P3 - P0)/3])
    res = least_squares(loss, guess, args=(points,), verbose=0)

    P1 = res.x[0:2]
    P2 = res.x[2:4]
    return np.array([P0, P1, P2, P3])

def sample_bezier(ctrl_pts, num=100):
    P0, P1, P2, P3 = ctrl_pts
    ts = np.linspace(0, 1, num)
    u = 1 - ts
    pts = (u**3)[:, None]*P0 + \
          3*(u**2 * ts)[:, None]*P1 + \
          3*(u * ts**2)[:, None]*P2 + \
          (ts**3)[:, None]*P3
    return pts  # shape (num, 2)

def get_base_width_pixels(mask):
    h, w = mask.shape
    last = mask[h-1, :]
    x_vals = np.where(last > 0)[0]
    if len(x_vals) < 2:
        raise RuntimeError("Base width not detected")
    width = x_vals[-1] - x_vals[0]
    center_x = x_vals.mean()
    return width, center_x, h - 1

def convert_pts_px_to_mm(pts_px, base_center_x, base_y, px_per_mm):
    pts_mm = []
    for x, y in pts_px:
        x_mm = (x - base_center_x) / px_per_mm
        y_mm = (base_y - y) / px_per_mm
        pts_mm.append([x_mm, y_mm])
    return np.array(pts_mm, dtype=float)

def parse_field_strength_from_filename(filepath):
    name = os.path.basename(filepath).split('.')[0]
    num = ''
    for ch in name:
        if ch.isdigit() or ch == '.':
            num += ch
        else:
            break
    if not num:
        raise ValueError(f"Couldn't parse number from filename: {filepath}")
    return float(num)

if __name__ == "__main__":
    output_dir = "output_scaled"
    os.makedirs(output_dir, exist_ok=True)

    calibration_data = {}

    for path in glob.glob(os.path.join("CRImages", "*.*")):
        try:
            img = load_and_crop(path)
            mask0 = threshold_blue_or_dark(img)
            mask = clean_mask(mask0)

            midline = compute_row_midpoints(mask)
            if midline.shape[0] < 2:
                raise RuntimeError("Not enough midline points")

            bez_ctrl = fit_cubic_bezier(midline)
            base_w_px, base_center_x, base_y = get_base_width_pixels(mask)
            px_per_mm = base_w_px / 4.0

            # Sample many points along the fitted curve (in 2D pixel space)
            sampled_px = sample_bezier(bez_ctrl, num=200)

            # Convert both control pts and sampled pts to mm
            bez_ctrl_mm = convert_pts_px_to_mm(bez_ctrl, base_center_x, base_y, px_per_mm)
            sampled_mm = convert_pts_px_to_mm(sampled_px, base_center_x, base_y, px_per_mm)

            B = parse_field_strength_from_filename(path)
            calibration_data[B] = {
                'bezier_control_points_mm': bez_ctrl_mm.tolist(),
                'centerline_mm': sampled_mm.tolist(),
                'base_width_mm': 4.0
            }

        except Exception as e:
            print("Failed for", path, ":", e)

    outpath = os.path.join(output_dir, "calibration_full_mm.json")
    with open(outpath, 'w') as f:
        json.dump(calibration_data, f, indent=2)

    print("Wrote calibration to", outpath)
