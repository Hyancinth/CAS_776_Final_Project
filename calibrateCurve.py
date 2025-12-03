import cv2
import numpy as np
import os
import glob
import json
from scipy.optimize import least_squares

# Loads image and crops the central region (default = center 50%)
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

# Quick mask to catch either bluish or generally dark regions — tuning may vary
def threshold_blue_or_dark(img_bgr, hue_min=90, hue_max=140, sat_min=30, val_min=50, dark_threshold=80):
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    blue_low = np.array([hue_min, sat_min, val_min], dtype=np.uint8)
    blue_high = np.array([hue_max, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv_img, blue_low, blue_high)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)

    combined = cv2.bitwise_or(blue_mask, dark_mask)
    return combined

# Just tidies up the binary mask — morphological ops
def clean_mask(mask, closing_kernel=(5,5), dilate_iter=1, erode_iter=1):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, closing_kernel)
    m = mask.copy()

    if dilate_iter:
        m = cv2.dilate(m, k, iterations=dilate_iter)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if erode_iter:
        m = cv2.erode(m, k, iterations=erode_iter)
    return m

# Extract midpoints of active regions, row-wise — this gives us a rough centerline
def compute_row_midpoints(mask, row_step=1):
    h, w = mask.shape
    pts = []
    for y in range(0, h, row_step):
        x_indices = np.where(mask[y, :] > 0)[0]
        if len(x_indices) >= 2:
            left = x_indices[0]
            right = x_indices[-1]
            center_x = (left + right) / 2.0
            pts.append((center_x, y))
    return np.array(pts, dtype=float)

# Cubic Bézier fitting using least squares – this is the core curve fitting logic
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

    guess = np.concatenate([P0 + (P3 - P0) / 3, P0 + 2 * (P3 - P0) / 3])
    res = least_squares(loss, guess, args=(points,), verbose=0)

    P1 = res.x[0:2]
    P2 = res.x[2:4]
    return np.array([P0, P1, P2, P3])

# Find the width of the shape at the very bottom row — assume this is known width in mm
def get_base_width_pixels(mask):
    h, w = mask.shape
    last_row = mask[h - 1, :]
    x_vals = np.where(last_row > 0)[0]
    if len(x_vals) < 2:
        raise RuntimeError("Base width not detected — bad bottom row?")
    width = x_vals[-1] - x_vals[0]
    center_x = x_vals.mean()
    return width, center_x, h - 1

# Convert from pixel coordinates to mm, using base-center as origin
def convert_control_pts_to_mm(control_pts_px, base_center_x, base_y, pixels_per_mm):
    points_mm = []
    for x, y in control_pts_px:
        x_mm = (x - base_center_x) / pixels_per_mm
        y_mm = (y - base_y) / pixels_per_mm
        points_mm.append([x_mm, y_mm])
    return points_mm

# Grab the numeric part of filename — used as the field strength
def parse_field_strength_from_filename(filepath):
    name = os.path.basename(filepath).split('.')[0]
    num_str = ''
    for ch in name:
        if ch.isdigit() or ch == '.':
            num_str += ch
        else:
            break
    if not num_str:
        raise ValueError(f"Couldn't parse number from filename: {filepath}")
    return float(num_str)

# Main processing block
if __name__ == "__main__":
    output_dir = "output_scaled"
    os.makedirs(output_dir, exist_ok=True)

    calibration_data = {}

    for path in glob.glob(os.path.join("CRImages", "*.*")):
        try:
            cropped = load_and_crop(path)
            mask_raw = threshold_blue_or_dark(cropped)
            mask = clean_mask(mask_raw)

            midline_pts = compute_row_midpoints(mask)
            if len(midline_pts) < 2:
                print(f"Skipping {path} — not enough midpoints to fit.")
                continue

            bez_ctrl_px = fit_cubic_bezier(midline_pts)

            base_width_px, base_center_x, base_y = get_base_width_pixels(mask)
            if base_width_px <= 0:
                raise RuntimeError("Invalid base width!")

            px_per_mm = base_width_px / 4.0  # 4 mm known reference
            bez_ctrl_mm = convert_control_pts_to_mm(bez_ctrl_px, base_center_x, base_y, px_per_mm)

            field_strength = parse_field_strength_from_filename(path)

            calibration_data[field_strength] = {
                'bezier_control_points_mm': bez_ctrl_mm,
                'base_width_mm': 4.0
            }

            # Note: could save debug overlays etc., but skipping for now

        except Exception as err:
            print("Error processing", path, ":", err)

    output_json = os.path.join(output_dir, "calibration_scaled_mm.json")
    with open(output_json, 'w') as f_out:
        json.dump(calibration_data, f_out, indent=2)

    print("Saved control points (mm, origin at base-center) to", output_json)
