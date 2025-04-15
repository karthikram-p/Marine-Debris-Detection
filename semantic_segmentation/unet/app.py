import os
import tempfile
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import rasterio
from zipfile import ZipFile
import glob

# Constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_CHANNELS = 11
OUTPUT_CLASSES = 11
HIDDEN_CHANNELS = 16
CHECKPOINT_PATH = "semantic_segmentation/unet/trained_models/best_model_marine_debris.pth"
SAMPLE_DATA_PATH = "semantic_segmentation/unet/sample_data"
QML_PATH = "semantic_segmentation/unet/mask_style.qml"

# Dummy mean/std (replace with actual ones)
bands_mean = np.ones(INPUT_CHANNELS)
bands_std = np.ones(INPUT_CHANNELS)

from unet_plus_plus import UNetPlusPlus


@st.cache_resource
def load_model():
    model = UNetPlusPlus(input_bands=INPUT_CHANNELS, output_classes=OUTPUT_CLASSES, hidden_channels=HIDDEN_CHANNELS)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def zip_sample_images(folder_path=SAMPLE_DATA_PATH, zip_name="sample_images.zip"):
    temp_zip_path = os.path.join(tempfile.gettempdir(), zip_name)
    with ZipFile(temp_zip_path, 'w') as zipf:
        for f in glob.glob(os.path.join(folder_path, "*.tif"))[:20]:
            zipf.write(f, os.path.basename(f))
    return temp_zip_path


def load_tiff_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name
    with rasterio.open(temp_path) as ds:
        img = ds.read()
        geo_transform = ds.transform
        projection = ds.crs.to_wkt() if ds.crs else None
    return img, geo_transform, projection, temp_path


def preprocess_image(img):
    img = img.astype(np.float32)
    img = np.where(np.isnan(img), np.tile(bands_mean[:, None, None], img.shape[1:]), img)
    img = (img - bands_mean[:, None, None]) / bands_std[:, None, None]
    return torch.tensor(img).unsqueeze(0).to(device)


def predict(tensor_img, model):
    with torch.no_grad():
        logits = model(tensor_img)
    probs = torch.softmax(logits, dim=1)
    return torch.argmax(probs, dim=1).squeeze(0).cpu().numpy() + 1


def save_prediction_as_tiff(mask, transform, projection):
    temp_path = tempfile.mktemp(suffix=".tif")
    profile = {
        'driver': 'GTiff',
        'height': mask.shape[0],
        'width': mask.shape[1],
        'count': 1,
        'dtype': 'uint8',
        'transform': transform,
        'crs': projection
    }
    with rasterio.open(temp_path, 'w', **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)
    os.chmod(temp_path, 0o666)
    return temp_path


# -------------------------
# UI Pages
# -------------------------

def show_home():
    st.title("🌊 Marine Debris Info & QGIS Guide")
    st.markdown("""
    <div style="padding: 10px 20px; background-color: #e0f7fa; border-radius: 8px;">
        <h3>🌐 What is Marine Debris?</h3>
        <p>Marine debris is human-made waste that ends up in oceans and waterways. It affects marine life, ecosystems, and human health.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧪 Classes Detected by the Model")
    st.dataframe({
        "Class ID": list(range(1, 16)),
        "Class Name": [
            "Marine Debris", "Dense Sargassum", "Sparse Sargassum", "Natural Organic Material", "Ship", "Clouds",
            "Marine Water", "Sediment-Laden Water", "Foam", "Turbid Water", "Shallow Water", "Waves",
            "Cloud Shadows", "Wakes", "Mixed Water"
        ]
    }, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🗺️ Viewing in QGIS")
    st.markdown("""
    To view your segmented output in QGIS with color labels:
    1. **Open** your `.tif` prediction output in QGIS.
    2. Right-click the layer → **Properties** → **Symbology**.
    3. Click **Style > Load Style...** and select the `.qml` file provided.
    4. 🎨 The mask will now be color-coded by class!

    > This allows quick interpretation of results directly in a GIS environment.
    """)
    st.info("Scroll to the bottom of the Segmentation page to download the QML file!")

def show_segmentation():
    st.title("🧠 Semantic Segmentation")

    with st.expander("📦 Download Sample TIFF Images"):
        st.write("Get up to 20 multi-band TIFF files to try out the model.")
        try:
            zip_path = zip_sample_images()
            with open(zip_path, "rb") as f:
                st.download_button("📥 Download Sample ZIP", f, "sample_images.zip", "application/zip")
        except Exception as e:
            st.error(str(e))

    uploaded_file = st.file_uploader("📂 Upload Multi-Band TIFF Image", type=["tiff", "tif"])

    if uploaded_file:
        st.subheader("🖼️ Processing Uploaded Image")
        img, transform, projection, temp_path = load_tiff_image(uploaded_file)

        if img.shape[0] != INPUT_CHANNELS:
            st.error(f"Expected {INPUT_CHANNELS} bands, but got {img.shape[0]}")
            return

        st.success("✅ Image loaded successfully!")

        model = load_model()
        tensor_img = preprocess_image(img)
        mask = predict(tensor_img, model)

        st.markdown("### 🎯 Segmentation Result")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots()
            ax.imshow(mask, cmap="tab20")
            ax.set_title("Predicted Mask")
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            tiff_out_path = save_prediction_as_tiff(mask, transform, projection)
            with open(tiff_out_path, "rb") as f:
                st.download_button("📥 Download Segmented TIFF", f, "segmented_output.tif", "image/tiff")

        # QML Download
        st.markdown("---")
        st.markdown("### 🎨 QGIS Color Mask Mapping")
        st.write("Apply class color styling in QGIS using this style file.")
        try:
            with open(QML_PATH, "rb") as f:
                st.download_button("🎨 Download QGIS Style (.qml)", f, "mask_style.qml", "text/xml")
        except FileNotFoundError:
            st.error("❌ QML file not found.")

        os.unlink(temp_path)
        os.unlink(tiff_out_path)

# Sidebar Navigation
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Select a page", ["🏠 Info & Guide", "🧪 Segmentation"])

if page == "🏠 Info & Guide":
    show_home()
else:
    show_segmentation()
