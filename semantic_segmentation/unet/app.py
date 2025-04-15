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

def zip_sample_images(folder_path="semantic_segmentation/unet/sample_data", zip_name="sample_images.zip"):
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

    with st.expander("📦 Download Sample TIFF Images"):
        st.write("Need data to test? Download samples of multi-band TIFF ocean images to try out the model.")
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

        # Add QGIS color mask mapping style (.qml)
        qml_path = os.path.join(os.path.dirname(__file__), "qgis_color_mask_mapping.qml")
        if os.path.exists(qml_path):
            st.markdown("### 🎨 Apply Colors in QGIS")
            st.markdown("Use this QML file to apply class-specific colors in QGIS after loading the segmented TIFF.")
            with open(qml_path, "rb") as style_file:
                st.download_button(
                    label="🎨 Download QGIS Style (.qml)",
                    data=style_file,
                    file_name="qgis_color_mask_mapping.qml",
                    mime="text/xml"
                )
        else:
            st.warning("⚠️ QML style file not found.")


        # Clean up the temporary file
        os.unlink(temp_file_path)
        os.unlink(tiff_path)

def about_page():
    st.title("📘 About Marine Debris Detection")
    st.markdown("---")

    st.markdown("""
    ### 🌊 What is Marine Debris?
    Marine debris refers to human-created waste that ends up in oceans, seas, lakes, and waterways—intentionally or unintentionally. Common types include:
    - Plastics (bags, bottles, microplastics)
    - Abandoned fishing gear (ghost nets)
    - Rubber, glass, metal, and textiles

    Most marine debris, especially plastic, **does not biodegrade**. Instead, it breaks down into smaller fragments called **microplastics**, which can be ingested by marine animals and even enter the human food chain.

    ---

    ### ♻️ Why Does It Matter?
    - 🐢 **Wildlife Impact:** Animals can become entangled or mistake debris for food.
    - 🪸 **Habitat Destruction:** Coral reefs and mangroves are especially vulnerable.
    - ⚓ **Navigation Hazards:** Floating debris can damage boats and ships.
    - 🧬 **Human Health Risk:** Microplastics have been detected in seafood and even drinking water.

    ---

    ### 🛰️ Role of Remote Sensing & Deep Learning
    **Remote sensing** using satellites and drones, when combined with **semantic segmentation models** like **U-Net++**, can:
    - Detect marine debris in multi-spectral satellite images
    - Classify pixels into debris types or regions
    - Monitor temporal trends in pollution
    - Guide clean-up missions and policy planning

    ---

    ### 🧠 About the Model
    This app uses a custom-trained **U-Net++ (nested U-Net)** model, fine-tuned on multi-band satellite TIFF images.

    **Input:** Multi-band (11-channel) imagery  
    **Output:** Pixel-level classification of debris types  
    **Framework:** PyTorch  

    ---

    ### 📦 Dataset Information

    #### 🧪 MARIDA Dataset
    The model is trained using the **MARIDA (MARine Debris Archive)** dataset — the **first open benchmark** for marine debris detection using **Sentinel-2 satellite imagery**.
    
    - Developed by **GEOMAR Helmholtz Centre for Ocean Research Kiel**
    - Provides labeled samples across **five classes**: marine debris, ships, clean water, natural organic material (NOM), and Sargassum
    - Includes **polygon annotations** and corresponding raster masks
    - Offers **1000+ annotations** from global coastal regions
    - Designed for training, validating, and benchmarking semantic segmentation models

    #### 🛰️ Other Sources
    - Sentinel-2 and PlanetScope imagery
    - Preprocessing steps: cloud masking, atmospheric correction, band normalization

    ---

    ### 🗺️ Visualizing Results with QGIS
    [**QGIS**](https://qgis.org/en/site/) is a powerful, open-source Geographic Information System for viewing, editing, and analyzing geospatial data.

    **How to visualize your results:**
    1. Download and install QGIS from the official site
    2. Open the `segmentation_output.tif` file
    3. Load the provided `.qml` file to apply class-based color mapping
    4. Overlay with basemaps, vector layers, or shipping lanes for deeper analysis

    **Use Cases in QGIS:**
    - Monitor marine debris hotspots over time
    - Integrate with environmental data layers
    - Generate maps for policy briefings or academic reports
    - Evaluate proximity to shipping routes or coastal cities

    ---

    ### 📘 Learn More

    - 🌐 [NOAA Marine Debris Program](https://marinedebris.noaa.gov/)
    - 🌐 [UNEP Clean Seas Campaign](https://www.cleanseas.org/)
    - 📊 [MARIDA Dataset on Zenodo](https://zenodo.org/record/7075396)
    - 📄 [MARIDA Paper – “MARIDA: A Benchmark for Marine Debris Detection from Sentinel-2 Remote Sensing Data”](https://arxiv.org/abs/2211.04768)
    - 📚 [QGIS Documentation](https://docs.qgis.org/)
    - 🛰️ [ESA Sentinel Hub – Open Access Earth Observation Data](https://www.sentinel-hub.com/)
    - 🧠 [U-Net++: Nested U-Net for Medical Image Segmentation](https://arxiv.org/abs/1807.10165)

    ---
    """)



# 🧭 Navigation
def main_router():
    page = st.sidebar.radio("🔍 Select Page", ["📘 About Marine Debris", "🏠 Segmentation App"])
    
    if page == "📘 About Marine Debris":
        about_page()
    elif page == "🏠 Segmentation App":
        main()

# 🔁 Run the router instead of directly calling main()
if __name__ == "__main__":
    main_router()

