import os
import tempfile
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import rasterio
from zipfile import ZipFile
import glob
from streamlit_lottie import st_lottie
import json
import requests

# Configure page settings
st.set_page_config(
    page_title="Marine Debris Detection 🌊",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_CHANNELS = 11
OUTPUT_CLASSES = 11
HIDDEN_CHANNELS = 16  
CHECKPOINT_PATH = "semantic_segmentation/unet/trained_models/best_model_marine_debris.pth"

from dataloader import bands_mean, bands_std
from unet_plus_plus import UNetPlusPlus


@st.cache_resource
def load_model():
    model = UNetPlusPlus(input_bands=INPUT_CHANNELS, output_classes=OUTPUT_CLASSES, hidden_channels=HIDDEN_CHANNELS)
    model_path = os.path.join(CHECKPOINT_PATH)

    if not os.path.exists(model_path):
        st.error(f"🚨 Model checkpoint not found at {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    st.success("✅ Model Loaded Successfully!")
    return model

def load_tiff_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    with rasterio.open(temp_file_path) as ds:
        image_data = ds.read()
        geo_transform = ds.transform
        projection = ds.crs.to_wkt() if ds.crs else None

    return image_data, geo_transform, projection, temp_file_path

def preprocess_image(image):
    img = np.array(image).astype(np.float32)
    nan_mask = np.isnan(img)
    mean_values = np.tile(bands_mean[:, None, None], (1, img.shape[1], img.shape[2]))
    img = np.where(nan_mask, mean_values, img)
    img = (img - bands_mean[:, None, None]) / bands_std[:, None, None]
    img_tensor = torch.tensor(img).unsqueeze(0).to(device)
    return img_tensor

def predict(image_tensor, model):
    with torch.no_grad():
        logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_mask = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()
    mapped_mask = predicted_mask + 1
    return mapped_mask

def save_prediction_as_tiff(predicted_mask, geo_transform, projection):
    temp_tiff_path = tempfile.mktemp(suffix=".tif")
    rows, cols = predicted_mask.shape
    profile = {
        'driver': 'GTiff',
        'height': rows,
        'width': cols,
        'count': 1,
        'dtype': 'uint8',
        'transform': geo_transform,
        'crs': projection
    }

    with rasterio.open(temp_tiff_path, 'w', **profile) as dst:
        dst.write(predicted_mask.astype(np.uint8), 1)

    os.chmod(temp_tiff_path, 0o666)
    return temp_tiff_path

def zip_sample_images(folder_path="semantic_segmentation/unet/sample_data", zip_name="sample_images.zip"):
    temp_zip_path = os.path.join(tempfile.gettempdir(), zip_name)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Sample folder '{folder_path}' does not exist.")
    
    with ZipFile(temp_zip_path, 'w') as zipf:
        tiff_files = glob.glob(os.path.join(folder_path, "*.tif"))
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in the folder '{folder_path}'.")
        
        for file_path in tiff_files[:20]:
            zipf.write(file_path, os.path.basename(file_path))
    
    return temp_zip_path

# Load Lottie animation from URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ----------------------------- Main UI Page ----------------------------- #
def main():
    lottie_debris = load_lottieurl("https://lottie.host/5b0bc72b-f96b-4cf6-b36f-2226469912e2/VdjUnAoHPP.json")
    st_lottie(lottie_debris, height=200, speed=1, key="debris-top")

    st.markdown("<h1 style='text-align: center; color: #1f3b4d;'>🌊 Marine Debris Segmentation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Detect marine debris from satellite images using a deep learning U-Net++ model.</p>", unsafe_allow_html=True)

    st.info("🔍 Using device: **{}**".format(device))

    with st.expander("📦 Download Sample TIFF Images"):
        st.markdown("Download multi-band TIFF ocean samples to test the model.")
        try:
            zip_path = zip_sample_images()
            with open(zip_path, "rb") as zf:
                st.download_button(
                    label="📥 Download Sample Images",
                    data=zf,
                    file_name="sample_images.zip",
                    mime="application/zip",
                    use_container_width=True
                )
        except FileNotFoundError as e:
            st.error(f"🚨 {e}")

    uploaded_tiff = st.file_uploader("📂 Upload a Multi-Band TIFF Image", type=["tiff", "tif"])

    if uploaded_tiff:
        st.markdown("### 🖼️ Uploaded Image Preview")
        image_data, geo_transform, projection, temp_file_path = load_tiff_image(uploaded_tiff)

        if image_data.shape[0] != INPUT_CHANNELS:
            st.error(f"🚨 Expected {INPUT_CHANNELS} bands, but got {image_data.shape[0]}")
            return

        model = load_model()
        if model is None:
            return

        image_tensor = preprocess_image(image_data)

        with st.spinner("🔍 Running model inference..."):
            predicted_mask = predict(image_tensor, model)
        st.success("✅ Segmentation Complete")

        st.subheader("📊 Predicted Segmentation Mask")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(predicted_mask, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

        tiff_path = save_prediction_as_tiff(predicted_mask, geo_transform, projection)
        with open(tiff_path, "rb") as file:
            st.download_button(
                label="📥 Download Segmentation Mask (TIFF)",
                data=file,
                file_name="segmentation_output.tif",
                mime="image/tiff",
                use_container_width=True
            )

        qml_path = os.path.join(os.path.dirname(__file__), "qgis_color_mask_mapping.qml")
        if os.path.exists(qml_path):
            with open(qml_path, "rb") as style_file:
                st.download_button(
                    label="🎨 Download QGIS Style File (.qml)",
                    data=style_file,
                    file_name="qgis_color_mask_mapping.qml",
                    mime="text/xml",
                    use_container_width=True
                )
        else:
            st.warning("⚠️ QGIS style file not found.")

        os.unlink(temp_file_path)
        os.unlink(tiff_path)


def about_page():
    st.markdown("<h1 style='color:#1f3b4d'>📘 About Marine Debris Detection</h1>", unsafe_allow_html=True)

    cols = st.columns([1, 2])
    with cols[0]:
        st.image("https://seahistory.org/wp-content/uploads/marine-debris.jpg", use_column_width=True)
    with cols[1]:
        st.markdown("""
        ### 🌊 What is Marine Debris?
        - Plastics (bags, bottles, microplastics)
        - Ghost fishing nets
        - Rubber, glass, metal, textiles
        - **Non-biodegradable** materials
        
        It poses serious risks to marine life, ecosystems, and even human health.
        """)

    st.markdown("---")
    st.image("https://marine-debris-site-s3fs.s3.us-west-1.amazonaws.com/s3fs-public/sea_turtle_entangled.jpg?VersionId=26WuPYKaZUe82w4GutIHwwK9adTByMaL", width=700)

    st.markdown("""
    ### 🧠 Why Deep Learning?
    - 🛰️ Detect marine debris via satellite images
    - 🧠 Segment and classify pixel data
    - 📈 Track pollution over time
    - 🧹 Plan cleanup & conservation missions
    """)

    st.image("https://www.esa.int/var/esa/storage/images/esa_multimedia/images/2015/03/sentinel-2/15292661-1-eng-GB/Sentinel-2_pillars.jpg")

    st.markdown("""
    ### 🧪 Dataset: MARIDA
    - Sentinel-2 based
    - Labels: debris, ships, clean water, Sargassum, NOM
    - Over 1000 annotated coastal samples
    """)

    st.markdown("---")
    st.markdown("""
    ### 🗺️ Visualize in QGIS
    - Load `segmentation_output.tif`
    - Apply `qgis_color_mask_mapping.qml`
    - Overlay with maps, ports, or zones
    """)

    st.image("https://upload.wikimedia.org/wikipedia/commons/5/5a/QGIS_Interface_Screenshot_with_Map_of_Median_Income_in_Houston_%282010%29.png")

    st.markdown("🔗 [More Resources](https://marinedebris.noaa.gov/) | [QGIS Docs](https://docs.qgis.org/)")


def main_router():
    st.sidebar.markdown("## 🔍 Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Segmentation App", "📘 About Marine Debris"])

    if page == "📘 About Marine Debris":
        about_page()
    elif page == "🏠 Segmentation App":
        main()


if __name__ == "__main__":
    main_router()
