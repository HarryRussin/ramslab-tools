import streamlit as st
import cv2
import os
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# --- CONFIGURATION ---
UPLOAD_DIR = "uploads"
CROP_DIR = "crops"
LABEL_FILE = "labeled_data.csv"
for d in [UPLOAD_DIR, CROP_DIR]:
    os.makedirs(d, exist_ok=True)

# Load YOLOv8 model (using 'nano' for speed)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt") 

model = load_model()

# --- APP UI ---
st.set_page_config(page_title="Ramslab Hockey Labeler", layout="wide")
st.title("Hockey Scoreboard Labeler")

# Sidebar: Video Upload
with st.sidebar:
    st.header("1. Data Source")
    uploaded_file = st.file_uploader("Upload Hockey Match", type=["mp4", "mov", "avi"])
    
    conf_threshold = st.slider("YOLO Confidence", 0.1, 1.0, 0.25)
    frame_skip = st.number_input("Frame Skip (Process every Nth frame)", min_value=1, value=30)

# --- STEP 1: PROCESSING ---
if uploaded_file and st.sidebar.button("Process Video"):
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    cap = cv2.VideoCapture(video_path)
    st.info("Detecting scoreboards... please wait.")
    
    count = 0
    saved_count = 0
    progress_bar = st.progress(0)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if count % frame_skip == 0:
            # Run YOLO detection
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            
            for i, r in enumerate(results[0].boxes):
                # We assume the scoreboard is usually 'tv' or 'laptop' class in COCO
                # or we just take the highest confidence detection if custom trained
                xyxy = r.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Crop and save
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_name = f"frame_{count}_obj_{i}.jpg"
                    cv2.imwrite(os.path.join(CROP_DIR, crop_name), crop)
                    saved_count += 1
            
            progress_bar.progress(min(count / total_frames, 1.0))
        count += 1
        
    cap.release()
    st.success(f"Done! Extracted {saved_count} potential scoreboards.")

# --- STEP 2: LABELING ---
st.divider()
st.header("2. Labeling Interface")

if os.path.exists(CROP_DIR) and len(os.listdir(CROP_DIR)) > 0:
    images = sorted([f for f in os.listdir(CROP_DIR) if f.endswith(".jpg")])
    
    # Session State to track index and data
    if 'idx' not in st.session_state: st.session_state.idx = 0
    if 'labels' not in st.session_state: st.session_state.labels = []

    if st.session_state.idx < len(images):
        curr_img_name = images[st.session_state.idx]
        img_path = os.path.join(CROP_DIR, curr_img_name)
        
        col_img, col_btns = st.columns([2, 1])
        
        with col_img:
            st.image(img_path, use_container_width=True)
            st.caption(f"Image {st.session_state.idx + 1} of {len(images)}: {curr_img_name}")

        with col_btns:
            st.write("### Classify this frame:")
            if st.button("✅ Clear (Visible)", use_container_width=True):
                st.session_state.labels.append({"filename": curr_img_name, "status": "clear"})
                st.session_state.idx += 1
                st.rerun()

            if st.button("Covered (Occluded)", use_container_width=True):
                st.session_state.labels.append({"filename": curr_img_name, "status": "covered"})
                st.session_state.idx += 1
                st.rerun()

            if st.button("Not a Scoreboard", use_container_width=True):
                st.session_state.labels.append({"filename": curr_img_name, "status": "invalid"})
                st.session_state.idx += 1
                st.rerun()
            
            if st.button("⏭ Skip"):
                st.session_state.idx += 1
                st.rerun()

    else:
        st.balloons()
        st.success("All frames labeled!")

    # --- STEP 3: EXPORT ---
    if st.session_state.labels:
        st.divider()
        df = pd.DataFrame(st.session_state.labels)
        st.write("### Current Progress")
        st.dataframe(df.tail(5))
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Labels CSV", csv, "hockey_labels.csv", "text/csv")
else:
    st.info("Upload and process a video to start labeling.")
