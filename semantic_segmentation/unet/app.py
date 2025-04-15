import os
import tempfile
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import rasterio
from zipfile import ZipFile
import glob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"🔍 Using device: {device}")

INPUT_CHANNELS = 11
OUTPUT_CLASSES = 11
HIDDEN_CHANNELS = 16  
CHECKPOINT_PATH = "semantic_segmentation/unet/trained_models/best_model_marine_debris.pth"

# Import band statistics
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
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    # Open the TIFF file with rasterio
    with rasterio.open(temp_file_path) as ds:
        # Read all bands into a numpy array (bands, height, width)
        image_data = ds.read()  # Shape: (bands, height, width)
        
        # Get geotransform and projection for output
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
    
    # Define the output profile
    profile = {
        'driver': 'GTiff',
        'height': rows,
        'width': cols,
        'count': 1,
        'dtype': 'uint8',
        'transform': geo_transform,
        'crs': projection
    }

    # Write the predicted mask to a GeoTIFF
    with rasterio.open(temp_tiff_path, 'w', **profile) as dst:
        dst.write(predicted_mask.astype(np.uint8), 1)

    # Ensure the file is readable
    os.chmod(temp_tiff_path, 0o666)
    return temp_tiff_path

def zip_sample_images(folder_path=r"Marine_Debris_Detection\semantic_segmentation\unet\sample_data", zip_name="sample_images.zip"):
    """
    Create a ZIP file containing sample TIFF images from the specified folder.
    """
    temp_zip_path = os.path.join(tempfile.gettempdir(), zip_name)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Sample folder '{folder_path}' does not exist.")
    
    with ZipFile(temp_zip_path, 'w') as zipf:
        tiff_files = glob.glob(os.path.join(folder_path, "*.tif"))
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in the folder '{folder_path}'.")
        
        for file_path in tiff_files[:20]:  # Limit to 20 files
            zipf.write(file_path, os.path.basename(file_path))
    
    return temp_zip_path


def main():
    st.title("🌊 Marine Debris Semantic Segmentation with U-Net")

    with st.expander("📦 Download Sample TIFF Images (20 max)"):
        st.write("Need data to test? Download up to 20 sample multi-band TIFF images to try out the model.")
        try:
            zip_path = zip_sample_images()
            with open(zip_path, "rb") as zf:
                st.download_button(
                    label="📥 Download Sample Images ZIP",
                    data=zf,
                    file_name="sample_images.zip",
                    mime="application/zip"
                )
        except FileNotFoundError as e:
            st.error(f"🚨 {e}")


    uploaded_tiff = st.file_uploader("📂 Upload Multi-Band TIFF Image", type=["tiff", "tif"])

    if uploaded_tiff:
        st.subheader("🖼️ Uploaded TIFF Image")
        image_data, geo_transform, projection, temp_file_path = load_tiff_image(uploaded_tiff)

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
            st.download_button(
                label="📥 Download Segmentation TIFF",
                data=file,
                file_name="segmentation_output.tif",
                mime="image/tiff"
            )

        # Clean up the temporary file
        os.unlink(temp_file_path)
        os.unlink(tiff_path)

if __name__ == "__main__":
    main()
