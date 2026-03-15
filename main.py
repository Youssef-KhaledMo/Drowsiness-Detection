"""
Drowsiness Detection System
----------------------------
Real-time driver drowsiness detection using dlib facial landmarks,
CLAHE preprocessing, and a Keras CNN model.

Usage:
    python main.py [--model MODEL] [--landmarks LANDMARKS]
                   [--threshold THRESHOLD] [--camera CAMERA]
                   [--frames FRAMES] [--ratio RATIO]
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from collections import deque
from pathlib import Path

import cv2
import dlib
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-platform audio
# ---------------------------------------------------------------------------

def _build_beep(frequency: int = 2500, duration_ms: int = 500):
    """Return a callable that plays an alert beep on any OS."""
    if platform.system() == "Windows":
        import winsound
        def beep() -> None:
            winsound.Beep(frequency, duration_ms)
    else:
        try:
            import pygame
            pygame.mixer.init(frequency=44100)
            # Generate a simple sine-wave tone buffer
            import math
            sample_rate = 44100
            n_samples = int(sample_rate * duration_ms / 1000)
            buf = np.array(
                [int(32767 * math.sin(2 * math.pi * frequency * t / sample_rate))
                 for t in range(n_samples)],
                dtype=np.int16,
            )
            buf = np.column_stack([buf, buf])  # stereo
            sound = pygame.sndarray.make_sound(buf)
            def beep() -> None:
                sound.play()
        except ImportError:
            logger.warning("pygame not found — audio alerts disabled.")
            def beep() -> None:
                pass
    return beep

# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------

DEFAULT_MODEL      = "drowsiness_detection_model.h5"
DEFAULT_LANDMARKS  = "shape_predictor_68_face_landmarks.dat"
DEFAULT_THRESHOLD  = 0.5
DEFAULT_CAMERA     = 0
DEFAULT_IMG_SIZE   = 80
DEFAULT_FRAMES     = 20      # rolling-buffer length
DEFAULT_RATIO      = 0.8     # fraction of drowsy frames needed to alarm
OFFSET_W_PCT       = 3       # horizontal eye-region expansion (%)
OFFSET_H_PCT       = 3       # vertical eye-region expansion (%)

# dlib 68-point landmark indices
LEFT_EYE_SLICE  = slice(36, 42)
RIGHT_EYE_SLICE = slice(42, 48)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time drowsiness detection")
    parser.add_argument("--model",      default=DEFAULT_MODEL,     help="Path to .h5 model")
    parser.add_argument("--landmarks",  default=DEFAULT_LANDMARKS, help="Path to dlib dat file")
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                        help="Prediction threshold — awake if ≥ threshold")
    parser.add_argument("--camera",     type=int,   default=DEFAULT_CAMERA,
                        help="Camera device index")
    parser.add_argument("--frames",     type=int,   default=DEFAULT_FRAMES,
                        help="Rolling buffer size for smoothing")
    parser.add_argument("--ratio",      type=float, default=DEFAULT_RATIO,
                        help="Drowsy-frame ratio to trigger alarm (0-1)")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalisation to a grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(gray)


def preprocess_eye(
    eye_crop: np.ndarray,
    img_size: int = DEFAULT_IMG_SIZE,
) -> np.ndarray:
    """
    Resize, normalise, and reshape an eye crop ready for model inference.

    Returns shape (img_size, img_size, 1), float32, values in [0, 1].
    """
    resized = cv2.resize(eye_crop, (img_size, img_size))
    normalised = resized.astype(np.float32) / 255.0
    return normalised.reshape(img_size, img_size, 1)

# ---------------------------------------------------------------------------
# Eye detection
# ---------------------------------------------------------------------------

def extract_eye_regions(
    gray: np.ndarray,
    detector: dlib.fhog_object_detector,
    predictor: dlib.shape_predictor,
    frame_shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """
    Detect faces and return bounding boxes for both eyes.

    Returns a list of (x1, y1, x2, y2) tuples, clamped to frame boundaries.
    """
    h, w = frame_shape[:2]
    faces = detector(gray, 0)
    regions: list[tuple[int, int, int, int]] = []

    for face in faces:
        landmarks = predictor(gray, face)
        pts = [(p.x, p.y) for p in landmarks.parts()]

        for eye_slice in (LEFT_EYE_SLICE, RIGHT_EYE_SLICE):
            eye_pts = np.array(pts[eye_slice])
            ex, ey, ew, eh = cv2.boundingRect(eye_pts)

            # Expand region by offset percentages
            off_w = int((OFFSET_W_PCT / 100) * (ex + ew))
            off_h = int((OFFSET_H_PCT / 100) * (ey + eh))
            x1 = max(0, ex - off_w * 2)
            y1 = max(0, ey - off_h * 3)
            x2 = min(w, ex + ew + off_w * 2)
            y2 = min(h, ey + eh + int(off_h * 3.5))

            regions.append((x1, y1, x2, y2))

    return regions

# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class DrowsinessDetector:
    """Encapsulates model, landmark predictor, and per-session state."""

    def __init__(
        self,
        model_path: str,
        landmark_path: str,
        threshold: float = DEFAULT_THRESHOLD,
        buffer_frames: int = DEFAULT_FRAMES,
        drowsy_ratio: float = DEFAULT_RATIO,
        img_size: int = DEFAULT_IMG_SIZE,
    ) -> None:
        self.threshold     = threshold
        self.drowsy_ratio  = drowsy_ratio
        self.img_size      = img_size
        self._history: deque[int] = deque(maxlen=buffer_frames)

        # Validate files before loading
        self._validate_files(model_path, landmark_path)

        logger.info("Loading facial landmark predictor …")
        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmark_path)

        logger.info("Loading drowsiness model …")
        self.model = self._load_model(model_path)
        logger.info("Detector ready.")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_files(*paths: str) -> None:
        for p in paths:
            if not Path(p).is_file():
                logger.error("Required file not found: %s", p)
                sys.exit(1)

    @staticmethod
    def _load_model(model_path: str):
        """Load Keras model via tf.keras (compatible with tensorflow==2.15)."""
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Per-frame inference
    # ------------------------------------------------------------------

    def predict_frame(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Run drowsiness detection on a single BGR frame.

        Returns:
            annotated_frame: frame with status overlay drawn.
            alarm:           True if drowsiness threshold exceeded.
        """
        # Convert to gray once and reuse for both detection and CLAHE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        regions = extract_eye_regions(gray, self.detector, self.predictor, frame.shape)

        # Apply preprocessing pipeline once on the enhanced gray image
        enhanced = apply_clahe(gray)

        eye_crops: list[np.ndarray] = []
        valid_regions: list[tuple[int, int, int, int]] = []

        for (x1, y1, x2, y2) in regions:
            crop = enhanced[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            eye_crops.append(preprocess_eye(crop, self.img_size))
            valid_regions.append((x1, y1, x2, y2))

        alarm = False

        if eye_crops:
            # Batch inference — single model call for all eyes in the frame
            batch = np.stack(eye_crops)          # (N, img_size, img_size, 1)
            preds = self.model(batch, training=False).numpy().flatten()

            awake_flags = [int(p >= self.threshold) for p in preds]

            # Draw eye rectangles
            for (x1, y1, x2, y2) in valid_regions:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Rolling buffer: 1 = awake for this frame, 0 = drowsy
            frame_awake = int(all(f == 1 for f in awake_flags))
            self._history.append(frame_awake)

            # Alarm when drowsy-frame ratio exceeds threshold
            if len(self._history) == self._history.maxlen:
                drowsy_ratio = 1.0 - (sum(self._history) / len(self._history))
                alarm = drowsy_ratio >= self.drowsy_ratio

            # Status overlay — drawn once per frame, after the loop
            is_currently_awake = all(f == 1 for f in awake_flags)
            label  = "Awake"  if is_currently_awake else "Sleepy"
            colour = (0, 255, 0) if is_currently_awake else (0, 0, 255)
            cv2.putText(frame, label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)

        if alarm:
            cv2.putText(frame, "! DROWSINESS ALERT !",
                        (10, frame.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        return frame, alarm

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    detector = DrowsinessDetector(
        model_path     = args.model,
        landmark_path  = args.landmarks,
        threshold      = args.threshold,
        buffer_frames  = args.frames,
        drowsy_ratio   = args.ratio,
    )

    beep = _build_beep()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Cannot open camera at index %d", args.camera)
        sys.exit(1)

    logger.info("Starting capture — press 'q' to quit.")

    frame_count  = 0
    alarm_active = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame capture failed — retrying …")
                continue

            frame_count += 1
            annotated, alarm = detector.predict_frame(frame)

            # Only beep on the rising edge of the alarm to avoid spam
            if alarm and not alarm_active:
                logger.warning("Drowsiness detected!")
                beep()
            alarm_active = alarm

            # FPS counter every 30 frames
            if frame_count % 30 == 0:
                logger.debug("Processed %d frames", frame_count)

            cv2.imshow("Drowsiness Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit requested by user.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources released.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run(parse_args())