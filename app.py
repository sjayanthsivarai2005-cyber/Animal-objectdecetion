"""
Animal Object Detection Web App — Streamlit + Ultralytics YOLOv8 + OpenCV.

Detects COCO animal classes (bird, cat, dog, horse, sheep, cow, elephant,
bear, zebra, giraffe) and draws bounding boxes with labels and confidence.
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image
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


@st.cache_resource
def load_yolo_model(model_name: str = "yolov8n.pt") -> YOLO:
    """Load and cache the YOLOv8 model (weights download on first use)."""
    return YOLO(model_name)


def bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to a BGR numpy array for OpenCV."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Use PNG, JPEG, or similar.")
    return img


def run_animal_detection(
    model: YOLO,
    bgr_image: np.ndarray,
    conf_threshold: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Run YOLO inference, keep only animal classes, draw boxes with OpenCV.

    Returns annotated BGR image and a list of detection dicts with keys:
    name, confidence, bbox (x1, y1, x2, y2).
    """
    # Ultralytics expects RGB or BGR; BGR from cv2 is fine
    results = model.predict(
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

        # Draw rectangle
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, thickness=2)

        # Label text: "cat 0.95"
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


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB for Streamlit / PIL display."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    st.set_page_config(
        page_title="Animal Object Detection App",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.title("Animal Object Detection App")
    st.markdown(
        "Upload an image and run **YOLOv8** to detect animals "
        "(bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe)."
    )

    uploaded = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        help="Supported: PNG, JPEG, WebP, BMP",
    )

    col1, col2 = st.columns(2)
    with col1:
        conf_min = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

    model = load_yolo_model("yolov8n.pt")

    if uploaded is not None:
        # Preview original (RGB for display)
        pil_preview = Image.open(io.BytesIO(uploaded.getvalue()))
        st.subheader("Uploaded image")
        st.image(pil_preview, use_container_width=True)

        if st.button("Run detection", type="primary"):
            uploaded.seek(0)
            raw = uploaded.read()
            try:
                bgr = bytes_to_bgr(raw)
            except ValueError as e:
                st.error(str(e))
                return

            with st.spinner("Running YOLOv8 inference..."):
                annotated_bgr, dets = run_animal_detection(
                    model, bgr, conf_threshold=conf_min
                )

            st.subheader("Detections")
            if not dets:
                st.info(
                    "No animals detected above the threshold. "
                    "Try lowering the confidence or use a clearer photo."
                )
            else:
                for j, d in enumerate(dets, start=1):
                    st.write(
                        f"{j}. **{d['name']}** — confidence: **{d['confidence']:.2%}**"
                    )

            st.subheader("Result (bounding boxes)")
            st.image(
                bgr_to_rgb(annotated_bgr),
                caption="Animals with bounding boxes and scores",
                use_container_width=True,
            )
    else:
        st.caption("Upload an image to begin.")


if __name__ == "__main__":
    main()
