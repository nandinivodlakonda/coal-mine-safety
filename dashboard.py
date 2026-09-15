import streamlit as st
import json
import os
from datetime import datetime
from PIL import Image
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
ALERTS_FILE = "alerts.json"
CONF_THRESHOLD = 0.5
LOCATION = "Mine Zone A"

VIOLATION_CLASSES = {
    "NO-Hardhat": "HELMET VIOLATION",
    "NO-Safety Vest": "SAFETY VEST VIOLATION",
}

HF_REPO_ID = "Hansung-Cho/yolov8-ppe-detection"
HF_FILENAME = "best.pt"

st.set_page_config(page_title="Coal Mine PPE Safety Dashboard", layout="wide")


# ---------------------------------------------------------
# MODEL LOADING (downloaded once, cached across reruns)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    model = YOLO(model_path)
    return model


# ---------------------------------------------------------
# ALERTS STORAGE HELPERS
# ---------------------------------------------------------
def load_alerts():
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def save_alerts(alerts):
    with open(ALERTS_FILE, "w") as f:
        json.dump(alerts, f, indent=2)


# ---------------------------------------------------------
# UI HEADER
# ---------------------------------------------------------
st.title("Coal Mine PPE Safety Dashboard")

model = load_model()

# ---------------------------------------------------------
# IMAGE UPLOAD + DETECTION
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image of a mine worker (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect PPE Violations"):
        with st.spinner("Running PPE detection..."):
            results = model.predict(np.array(image), conf=CONF_THRESHOLD)
            result = results[0]

            # Annotated image (YOLO draws boxes + labels for us)
            annotated_array = result.plot()  # returns BGR numpy array
            annotated_image = Image.fromarray(annotated_array[:, :, ::-1])

            st.image(
                annotated_image,
                caption="Detection Result",
                use_container_width=True,
            )

            # Collect violations from this image
            new_alerts = []
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                if class_name in VIOLATION_CLASSES and confidence >= CONF_THRESHOLD:
                    new_alerts.append(
                        {
                            "alert_type": VIOLATION_CLASSES[class_name],
                            "confidence": round(confidence, 2),
                            "timestamp": timestamp,
                            "location": LOCATION,
                        }
                    )

            if new_alerts:
                st.error("⚠️ PPE VIOLATION DETECTED")
                for alert in new_alerts:
                    st.write(
                        f"**{alert['alert_type']}** — "
                        f"Confidence: {alert['confidence']} — "
                        f"Time: {alert['timestamp']} — "
                        f"Location: {alert['location']}"
                    )

                all_alerts = load_alerts()
                all_alerts.extend(new_alerts)
                save_alerts(all_alerts)
            else:
                st.success("No PPE violation detected")

st.divider()

# ---------------------------------------------------------
# DASHBOARD METRICS + RECENT ALERTS
# ---------------------------------------------------------
st.subheader("Dashboard Overview")

alerts = load_alerts()

if alerts:
    total_alerts = len(alerts)
    helmet_violations = sum(1 for a in alerts if a["alert_type"] == "HELMET VIOLATION")
    vest_violations = sum(
        1 for a in alerts if a["alert_type"] == "SAFETY VEST VIOLATION"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Alerts", total_alerts)
    col2.metric("Helmet Violations", helmet_violations)
    col3.metric("Safety Vest Violations", vest_violations)

    st.subheader("Recent Alerts")
    df = pd.DataFrame(alerts)
    st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
else:
    st.info("No alerts found. Upload an image and run detection to generate alerts.")
