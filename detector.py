"""
YOLOv8-based animal detection with OpenCV drawing.

COCO animal classes only: bird, cat, dog, horse, sheep, cow, elephant,
bear, zebra, giraffe.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# COCO animal class names (YOLOv8 pretrained on COCO includes these)
# ---------------------------------------------------------------------------
ANIMAL_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    }
)

# BGR colors for boxes (visible on most images)
BOX_COLOR = (0, 200, 100)
TEXT_BG_COLOR = (0, 200, 100)
TEXT_COLOR = (255, 255, 255)


def bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to a BGR numpy array for OpenCV."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Use PNG, JPEG, or similar.")
    return img


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB for Streamlit / PIL display."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class AnimalDetector:
    """
    Wraps Ultralytics YOLOv8: loads weights once, runs inference, filters to
    animal classes, draws bounding boxes with OpenCV.
    """

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        # Weights download automatically on first use if not cached.
        self._model = YOLO(model_name)

    @property
    def model(self) -> YOLO:
        """Expose underlying YOLO for advanced use if needed."""
        return self._model

    def detect(
        self,
        bgr_image: np.ndarray,
        conf_threshold: float,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Run detection on a BGR image.

        Returns annotated BGR image and a list of dicts with keys:
        name, confidence, bbox (x1, y1, x2, y2).
        """
        results = self._model.predict(
            source=bgr_image,
            conf=conf_threshold,
            verbose=False,
        )
        if not results:
            return bgr_image.copy(), []

        r0 = results[0]
        names: dict[int, str] = r0.names or {}
        out = bgr_image.copy()
        detections: list[dict[str, Any]] = []

        boxes = r0.boxes
        if boxes is None or len(boxes) == 0:
            return out, []

        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(box.cls[0].item())
            label = names.get(cls_id, str(cls_id))
            if label not in ANIMAL_CLASS_NAMES:
                continue

            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            detections.append(
                {
                    "name": label,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                }
            )

            cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, thickness=2)

            text = f"{label} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            ty1 = max(0, y1 - th - 8)
            cv2.rectangle(
                out,
                (x1, ty1),
                (x1 + tw + 4, ty1 + th + baseline + 4),
                TEXT_BG_COLOR,
                thickness=-1,
            )
            cv2.putText(
                out,
                text,
                (x1 + 2, ty1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

        return out, detections
