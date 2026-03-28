# Animal Object Detection App

A simple **Streamlit** web app that detects animals in uploaded images using **Ultralytics YOLOv8**, **OpenCV** for drawing, and **NumPy** / **Pillow** for image handling.

## Detected classes

The app uses the COCO-pretrained YOLOv8 model and keeps only these animal classes: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe.

## Installation

1. **Python 3.9+** recommended.

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies using the **same Python** you will use to run the app:

   ```powershell
   python -m pip install -r requirements.txt
   ```

   Required libraries: `streamlit`, `opencv-python`, `ultralytics`, `numpy`, `Pillow`.

   If you see **Permission denied** when installing into the system Python folder, install into your user profile instead:

   ```powershell
   python -m pip install --user -r requirements.txt
   ```

   That puts packages under `%APPDATA%\Python\Python313\site-packages` (path varies by Python version).

4. On first run, YOLOv8 will download `yolov8n.pt` weights automatically.

## Run locally

From the project directory:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Usage

1. Upload an image (PNG, JPEG, WebP, or BMP).
2. Adjust the **confidence threshold** if needed.
3. Click **Run detection**.
4. Review the list of detected animals with scores and the image with bounding boxes.

## Note

Do not name your script `streamlit.py` — it conflicts with the `streamlit` package imports.

## Troubleshooting

### `ModuleNotFoundError: No module named 'ultralytics'`

You have not installed dependencies for the Python interpreter you are using. Fix it with the same executable you use to run the app, for example:

```powershell
& "C:/Users/sjaya/AppData/Local/Programs/Python/Python313/python.exe" -m pip install --user -r requirements.txt
```

Then start the app with that same Python (or activate the venv where you installed):

```powershell
streamlit run app.py
```

### Run the app with Streamlit

Use `streamlit run app.py` to start the web UI. Running `python app.py` may not start the server correctly and still requires all packages to be installed.

### `pip install` permission errors on Windows

Use `python -m pip install --user ...`, run the terminal **as Administrator**, or use a **virtual environment** in a folder you own (recommended for long-term use).
