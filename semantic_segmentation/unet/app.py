import os
import tempfile
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from osgeo import gdal
from unet_plus_plus import UNetPlusPlus
from dataloader import bands_mean, bands_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"🔍 Using device: {device}")

INPUT_CHANNELS = 11
OUTPUT_CLASSES = 11
HIDDEN_CHANNELS = 16  
CHECKPOINT_PATH = "trained_models/best_model_marine_debris.pth"

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
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name  
    ds = gdal.Open(temp_file_path)
    if ds is None:
        raise ValueError("Could not open the TIFF file with GDAL.")
    image_data = ds.ReadAsArray()
    return image_data, ds.GetGeoTransform(), ds.GetProjection()

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
    temp_tiff_path = "predicted_output.tif"
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = predicted_mask.shape
    dataset = driver.Create(temp_tiff_path, cols, rows, 1, gdal.GDT_Byte)

    if geo_transform:
        dataset.SetGeoTransform(geo_transform)
    if projection:
        dataset.SetProjection(projection)

    dataset.GetRasterBand(1).WriteArray(predicted_mask.astype(np.uint8))
    dataset.FlushCache()
    dataset = None
    return temp_tiff_path

def main():
    st.title("🌊 Marine Debris Semantic Segmentation with U-Net")

    uploaded_tiff = st.file_uploader("📂 Upload Multi-Band TIFF Image", type=["tiff", "tif"])

    if uploaded_tiff:
        st.subheader("🖼️ Uploaded TIFF Image")
        image_data, geo_transform, projection = load_tiff_image(uploaded_tiff)

        if image_data.shape[0] != INPUT_CHANNELS:
            st.error(f"🚨 Expected {INPUT_CHANNELS} bands, but got {image_data.shape[0]}")
            return

        model = load_model()
        if model is None:
            return

        image_tensor = preprocess_image(image_data)

        st.subheader("🧠 Running Model...")
        predicted_mask = predict(image_tensor, model)
        st.success("✅ Prediction Complete!")

        st.subheader("🖥️ Predicted Segmentation Mask (Grayscale)")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(predicted_mask, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

        tiff_path = save_prediction_as_tiff(predicted_mask, geo_transform, projection)
        with open(tiff_path, "rb") as file:
            st.download_button(label="📥 Download Segmentation TIFF", data=file, file_name="segmentation_output.tif", mime="image/tiff")

if __name__ == "__main__":
    main()
