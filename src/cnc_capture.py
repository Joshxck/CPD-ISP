#!/usr/bin/env python3
"""
CNC Bed Webcam Capture
Captures high-quality images from a cheap webcam using frame averaging,
manual camera controls, and optional lens distortion correction.
"""

import cv2
import numpy as np
import argparse
import time
import os
from pathlib import Path


# ─── Camera Parameters ────────────────────────────────────────────────────────

DEFAULTS = {
    "device":       0,        # Camera index (try 0, 1, 2 if wrong camera)
    "frames":       30,       # Frames to average (more = less noise)
    "width":        1280,     # Capture width  (set to your webcam's max)
    "height":       720,      # Capture height (set to your webcam's max)
    "exposure":     -6,       # Manual exposure (negative = darker; tune this)
    "gain":         0,        # Sensor gain — keep at 0, use more light instead
    "sharpness":    0,        # In-camera sharpening — off, we do it in post
    "contrast":     0,        # In-camera contrast — off
    "brightness":   128,      # Midpoint; adjust if image is too dark/bright
    "warmup":       60,      # Give auto WB time to settle
    "output_dir":   ".",      # Where to save images
    "calibration":  None,     # Path to calibration .npz file (optional)
    "post_sharpen": True,     # Apply unsharp mask after averaging
}


# ─── Camera Setup ─────────────────────────────────────────────────────────────

def open_camera(device: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)  # Windows DirectShow
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)             # Fallback to default backend
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Don't force MJPEG — on Windows/DirectShow it can return raw YUV
    # which OpenCV misreads as BGR, causing the orange/blue colour shift.

    return cap


def save_settings(cap: cv2.VideoCapture) -> dict:
    """Read and save current camera settings so we can restore them on exit."""
    return {
        "auto_exposure":  cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        "exposure":       cap.get(cv2.CAP_PROP_EXPOSURE),
        "gain":           cap.get(cv2.CAP_PROP_GAIN),
        "auto_wb":        cap.get(cv2.CAP_PROP_AUTO_WB),
        "sharpness":      cap.get(cv2.CAP_PROP_SHARPNESS),
        "contrast":       cap.get(cv2.CAP_PROP_CONTRAST),
        "brightness":     cap.get(cv2.CAP_PROP_BRIGHTNESS),
    }


def restore_settings(cap: cv2.VideoCapture, original: dict):
    """Put the camera back exactly as we found it."""
    print("Restoring original camera settings...", end=" ", flush=True)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,  original["auto_exposure"])
    cap.set(cv2.CAP_PROP_EXPOSURE,       original["exposure"])
    cap.set(cv2.CAP_PROP_GAIN,           original["gain"])
    cap.set(cv2.CAP_PROP_AUTO_WB,        original["auto_wb"])
    cap.set(cv2.CAP_PROP_SHARPNESS,      original["sharpness"])
    cap.set(cv2.CAP_PROP_CONTRAST,       original["contrast"])
    cap.set(cv2.CAP_PROP_BRIGHTNESS,     original["brightness"])
    print("done.")


def apply_manual_controls(cap: cv2.VideoCapture, cfg: dict):
    """Lock exposure, gain, WB — disable all auto adjustments."""

    # On Windows/DirectShow, AUTO_EXPOSURE values are:
    #   0.25 (or 1) = manual, 0.75 (or 3) = auto
    # We try both common manual values and check which one actually changes exposure
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    time.sleep(0.2)
    cap.set(cv2.CAP_PROP_EXPOSURE, cfg["exposure"])

    # Gain to minimum
    cap.set(cv2.CAP_PROP_GAIN, cfg["gain"])

    # Let auto white balance run
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    # Flatten in-camera processing
    cap.set(cv2.CAP_PROP_SHARPNESS,  cfg["sharpness"])
    cap.set(cv2.CAP_PROP_CONTRAST,   cfg["contrast"])
    cap.set(cv2.CAP_PROP_BRIGHTNESS, cfg["brightness"])


def print_actual_settings(cap: cv2.VideoCapture):
    """Print what the driver actually accepted (may differ from what we asked)."""
    props = {
        "Resolution":  f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                       f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
        "Exposure":    cap.get(cv2.CAP_PROP_EXPOSURE),
        "Gain":        cap.get(cv2.CAP_PROP_GAIN),
        "Brightness":  cap.get(cv2.CAP_PROP_BRIGHTNESS),
        "Contrast":    cap.get(cv2.CAP_PROP_CONTRAST),
        "Sharpness":   cap.get(cv2.CAP_PROP_SHARPNESS),
        "FPS":         cap.get(cv2.CAP_PROP_FPS),
    }
    print("\n── Actual camera settings ──────────────────")
    for k, v in props.items():
        print(f"  {k:<12}: {v}")
    print("────────────────────────────────────────────\n")


# ─── Frame Averaging ──────────────────────────────────────────────────────────

def warmup(cap: cv2.VideoCapture, n: int):
    """Discard frames so the sensor stabilises before we start capturing."""
    print(f"Warming up ({n} frames)...", end=" ", flush=True)
    for _ in range(n):
        cap.read()
    print("done.")


def detect_color_format(cap: cv2.VideoCapture):
    """
    Capture a single frame and save versions in BGR, RGB, and BGRA
    so you can visually pick the correct one.
    """
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not grab frame for format detection")

    cv2.imwrite("format_bgr.png", frame)                                    # OpenCV default
    cv2.imwrite("format_rgb.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))   # If camera sends RGB
    cv2.imwrite("format_bgra.png", cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)) # If camera sends BGRA

    print("\nSaved format_bgr.png, format_rgb.png, format_bgra.png")
    print("Open them and find which looks correct, then pass --format rgb/bgr/bgra\n")


def convert_frame(frame: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if fmt == "bgra":
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame  # bgr — OpenCV native, no conversion needed


def capture_averaged(cap: cv2.VideoCapture, n_frames: int, fmt: str = "bgr") -> np.ndarray:
    """
    Capture n_frames and return their per-pixel mean.
    Averaging cancels random (Gaussian) noise — SNR improves by √n.
    """
    print(f"Capturing {n_frames} frames to average...", end=" ", flush=True)

    # Collect raw frames first, no processing
    frames = []
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Frame grab failed at frame {i}")
        frames.append(frame)

    # Save the very first raw frame — should look identical to probe output
    cv2.imwrite("debug_raw_frame.png", frames[0])
    print(f"\n  debug_raw_frame.png saved (raw, no processing)")

    # Average
    accumulator = np.zeros_like(frames[0], dtype=np.float32)
    for frame in frames:
        accumulator += frame.astype(np.float32)
    averaged = (accumulator / n_frames).astype(np.uint8)

    # Save averaged before colour conversion
    cv2.imwrite("debug_averaged_raw.png", averaged)
    print(f"  debug_averaged_raw.png saved (averaged, no colour conversion)")

    # Apply colour conversion
    result = convert_frame(averaged, fmt)
    cv2.imwrite("debug_averaged_converted.png", result)
    print(f"  debug_averaged_converted.png saved (after colour conversion)")

    print("done.")
    return result


# ─── Post-Processing ──────────────────────────────────────────────────────────

def unsharp_mask(img: np.ndarray,
                 radius: float = 1.5,
                 strength: float = 0.8) -> np.ndarray:
    """
    Gentle unsharp mask — enhances real edges without amplifying noise.
    radius:   blur sigma (smaller = finer detail)
    strength: blend factor (0=no effect, 1=full; keep ≤1 for realism)
    """
    k = int(radius * 6) | 1          # kernel size must be odd
    blurred = cv2.GaussianBlur(img, (k, k), radius)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)


def correct_distortion(img: np.ndarray,
                       calib_file: str) -> np.ndarray:
    """
    Undistort using a calibration matrix produced by calibrate.py.
    Removes barrel/pincushion distortion from cheap wide-angle lenses.
    """
    data = np.load(calib_file)
    mtx  = data["camera_matrix"]
    dist = data["dist_coeffs"]
    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted   = cv2.undistort(img, mtx, dist, None, new_mtx)
    x, y, rw, rh  = roi
    return undistorted[y:y+rh, x:x+rw] if all([rw, rh]) else undistorted


# ─── Calibration Helper ───────────────────────────────────────────────────────

def run_calibration(device: int, width: int, height: int,
                    cols: int = 9, rows: int = 6,
                    n_samples: int = 15,
                    output: str = "calibration.npz"):
    """
    Interactive lens calibration using a printed checkerboard.
    Print a checkerboard (e.g. https://calib.io/pages/camera-calibration-pattern-generator)
    and hold it at different angles/positions in front of the camera.
    Press SPACE to capture a sample, Q to quit early.
    """
    cap = open_camera(device, width, height)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    obj_points, img_points = [], []
    print(f"Calibration mode — need {n_samples} good samples.")
    print("  SPACE = capture  |  Q = finish early\n")

    while len(obj_points) < n_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, (cols, rows), corners, found)
            cv2.putText(display, "SPACE to capture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No board found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(display, f"Samples: {len(obj_points)}/{n_samples}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and found:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            obj_points.append(objp)
            img_points.append(corners2)
            print(f"  Captured sample {len(obj_points)}/{n_samples}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points) < 4:
        print("Not enough samples — calibration aborted.")
        return

    print("Computing calibration...", end=" ")
    h, w = gray.shape
    _, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    np.savez(output, camera_matrix=mtx, dist_coeffs=dist)
    print(f"done. Saved to {output}")


# ─── Main Capture Flow ────────────────────────────────────────────────────────

def capture(cfg: dict) -> str:
    cap = cv2.VideoCapture(cfg["device"], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    time.sleep(0.2)
    cap.set(cv2.CAP_PROP_EXPOSURE, cfg["exposure"])

    # Settle exactly like the probe does — 10 frames, no more
    print("Settling (10 frames)...", end=" ", flush=True)
    for _ in range(10):
        cap.read()
    print("done.")

    image = capture_averaged(cap, cfg["frames"], cfg["fmt"])
    cap.release()

    # Undistort first (before sharpening)
    if cfg["calibration"] and os.path.exists(cfg["calibration"]):
        print("Applying lens distortion correction...", end=" ", flush=True)
        image = correct_distortion(image, cfg["calibration"])
        print("done.")

    # Gentle unsharp mask
    if cfg["post_sharpen"]:
        print("Applying unsharp mask...", end=" ", flush=True)
        image = unsharp_mask(image, radius=1.5, strength=0.7)
        print("done.")

    # Save as lossless PNG
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(cfg["output_dir"]) / f"cnc_{ts}.png"
    cv2.imwrite(str(out_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"\n✓ Saved: {out_path}")
    return str(out_path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CNC webcam high-quality capture")
    p.add_argument("--device",      type=int,   default=DEFAULTS["device"],
                   help="Camera device index")
    p.add_argument("--frames",      type=int,   default=DEFAULTS["frames"],
                   help="Frames to average (default: 30)")
    p.add_argument("--width",       type=int,   default=DEFAULTS["width"])
    p.add_argument("--height",      type=int,   default=DEFAULTS["height"])
    p.add_argument("--exposure",    type=float, default=DEFAULTS["exposure"],
                   help="Manual exposure value (negative integers, e.g. -6)")
    p.add_argument("--gain",        type=float, default=DEFAULTS["gain"])
    p.add_argument("--output-dir",  type=str,   default=DEFAULTS["output_dir"],
                   help="Directory to save images")
    p.add_argument("--calibration", type=str,   default=DEFAULTS["calibration"],
                   help="Path to calibration .npz file for undistortion")
    p.add_argument("--no-sharpen",  action="store_true",
                   help="Skip post-processing unsharp mask")
    p.add_argument("--format",      type=str,   default="bgr",
                   choices=["bgr", "rgb", "bgra"],
                   help="Color format the camera outputs (default: bgr)")
    p.add_argument("--detect-format", action="store_true",
                   help="Save test images in all formats so you can pick the right one")
    p.add_argument("--calibrate",   action="store_true",
                   help="Run interactive lens calibration instead of capturing")
    args = p.parse_args()

    if args.calibrate:
        run_calibration(args.device, args.width, args.height)
        return

    if args.detect_format:
        cap = open_camera(args.device, args.width, args.height)
        warmup(cap, 10)
        detect_color_format(cap)
        cap.release()
        return

    cfg = {
        "device":       args.device,
        "frames":       args.frames,
        "width":        args.width,
        "height":       args.height,
        "exposure":     args.exposure,
        "gain":         args.gain,
        "sharpness":    DEFAULTS["sharpness"],
        "contrast":     DEFAULTS["contrast"],
        "brightness":   DEFAULTS["brightness"],
        "warmup":       DEFAULTS["warmup"],
        "output_dir":   args.output_dir,
        "calibration":  args.calibration,
        "fmt":          args.format,
        "post_sharpen": not args.no_sharpen,
    }

    os.makedirs(cfg["output_dir"], exist_ok=True)
    capture(cfg)


if __name__ == "__main__":
    main()