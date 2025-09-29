import os
# ---- Keep these BEFORE importing TF/Keras ----
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras import mixed_precision
from transformers import TFViTModel  # needed for deserialization

# Runtime policy
mixed_precision.set_global_policy("float32")

# App configuration
st.set_page_config(page_title = "Alzheimer Prediction", page_icon ="🧠", layout = "centered")
if "raw_pred" not in st.session_state:
    st.session_state.raw_pred = None
if "last_file" not in st.session_state:
    st.session_state.last_file = None
if "history" not in st.session_state:
    st.session_state.history = []


st.title("Alzheimer's Prediction App")

st.divider()

st.markdown("""
### What is Alzheimer’s disease?
Alzheimer’s is a disease that slowly affects **memory, thinking, and everyday tasks**.  
It often starts with small memory slips and can get worse over time, making planning and
communication harder. Getting older raises the risk, but it’s **not** just “normal aging.”
Doctors diagnose Alzheimer’s using your medical history, memory tests, and sometimes brain
scans or lab tests.
            
### Why retina images?
The retina is a small but important part of the nervous system. It develops from the same tissue as the brain and is made up of many nerve cells and blood vessels. Because of this close connection, changes in the retina can sometimes reflect changes happening in the brain.

Studies suggest that Alzheimer’s disease may be linked to signs in the retina, such as a loss of cells, thinning of certain layers, or changes in blood flow. Looking at the retina is much simpler than doing a brain scan, since it is non-invasive, quick, and widely available. This makes retinal imaging a useful way to study possible early signs of Alzheimer’s.
""")

# Optional tabs for more detail
tabs = st.tabs(["How the model works", "Limitations"])

with tabs[0]:
    st.write(
        "- **Preprocessing** - The retina image is converted to grayscale and resized to 50x50.\n\n"
        "- **Model** - Vision Image Transformer trained on retinal images. the ViT slices the images into smaller patches and learns how these patches relate to one another. Allowing the model to detect broader patterns within the image.\n\n"
        "- **Choosing the threshold** - The threshold is a cut-off value that decides whether the "
        "prediction is labeled as Alzheimer’s or Non-Alzheimer’s. A lower threshold makes the model "
        "more sensitive (fewer missed cases, but more false positives), while a higher threshold makes "
        "the model more specific (fewer false alarms, but some true cases may be missed). Users can "
        "adjust this slider depending on whether they want to prioritize sensitivity or specificity."
    )

with tabs[1]:
    st.write(
        "- This is **not** a substitute for clinical assessment.\n"
        "- Performance depends on the training data and image quality; real-world cases may differ.\n"
        "- Retinal images may be influenced by many other factors such as diabetes, age, hypertension, etc.\n"
    )

st.divider()
st.write("### **Get started by uploading an image of a retina below**")

# Loading the model
MODEL_FOLDERS = {
    "Model 1" : {
        "path": "vit_pretrained_fine_tuning",
        "input": {"size" : (50,50), "channels":1},
        "mode": "gray"
    },
    "Model 384" :{
        "path": "vit_pretrained_feature_extractor_384",
        "input" : {"size" : (50,50), "channels": 1},
        "mode": "gray"    
    },

    "Model 224" :{
        "path": "vit_pretrained_fine_tuning_224",
        "input" : {"size" : (50,50), "channels" : 1},
        "mode" : "gray"
    },

    "Model scratch" : {
        "path": "vit_scratch",
        "input": {"size": (50,50), "channels" : 1},
        "mode" : "gray"
    },

}

@st.cache_resource(show_spinner=True)
def load_model(key):
    config = MODEL_FOLDERS[key]
    model = tf.keras.models.load_model(
        config["path"], 
        custom_objects = {"TFViTModel": TFViTModel},
        compile = False,
        safe_mode = False
    )
    return model

# ---- Inference helpers ----
TARGET_H, TARGET_W, TARGET_C = 50, 50, 1
CLASS_NAMES = ["Non-Alzheimer's", "Alzheimer's"]

def preprocess(pil_img: Image.Image):
    img = np.array(pil_img.convert("L"), dtype=np.float32)  # grayscale
    img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR)
    if img.max() > 0:
        img = img / img.max()
    img = img[..., None]  # add channel dimension
    return np.expand_dims(img, axis=0)  # add batch dimension


def postprocess(pred, threshold=0.5):
    # Model outputs a single sigmoid neuron
    p = float(1 / (1 + np.exp(-pred.squeeze())))
    label_idx = int(p >= threshold)
    confid = p if label_idx == 1 else 1.0 - p
    label = CLASS_NAMES[label_idx]
    probs = {"Alzheimer's": p, "Non-Alzheimer's": 1 - p}
    return label, confid, probs

# UI
model_key = st.selectbox("Choose model", list(MODEL_FOLDERS.keys()))
threshold = st.slider("Decision threshold for Alzheimer's", 0.0, 1.0, 0.5, 0.01)
uploaded = st.file_uploader("Upload retinal image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])


col1, col2 = st.columns([1, 1])

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    col1.image(pil_img, caption="Uploaded image", use_container_width=True)

    # buttons
    c1, c2 = col1.columns([1,1])
    run_clicked = c1.button("Predict", use_container_width=True)
    reset_clicked = c2.button("Reset", use_container_width=True)

    if reset_clicked:
        st.session_state.raw_pred = None

    if run_clicked:
        with st.spinner("Running inference..."):
            cfg = MODEL_FOLDERS[model_key]
            model = load_model(model_key)
            x = preprocess(pil_img)
            pred = model.predict(x, verbose=0)
            st.session_state.raw_pred = pred

            lbl, conf, pr = postprocess(st.session_state.raw_pred, threshold=threshold)
            st.session_state.history.append({
                "file": uploaded.name,
                "model": model_key,
                "label": lbl,
                "confidence": str(round(conf*100,1)) + "%",
                "probs": pr
            })
        st.toast("Done")

if st.session_state.raw_pred is not None:
    lbl, conf, pr = postprocess(np.array(st.session_state.raw_pred), threshold=threshold)

    col2.metric("Prediction", lbl, delta=f"{conf*100:.1f}% conf")
    st.write("Class probs:")
    st.write(pr)

    with st.expander("Details"):
        st.write(f"Input resized to {TARGET_W}x{TARGET_H} grayscale "
                 f"then passed to ViT backbone.")
else:
    if uploaded is not None:
        col2.info("Click Predict to run model")

# history
if st.session_state.history:
    st.subheader("Prediction History")
    for i, h in enumerate(reversed(st.session_state.history), 1):
        # show model in the header
        header = f"{i}. {h['file']} — {h.get('model','(unknown model)')} — {h['label']} ({h['confidence']})"
        with st.expander(header):
            st.write(f"**Model used:** {h.get('model','N/A')}")
            st.json(h['probs'])

