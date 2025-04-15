import os
import tempfile
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import rasterio
from zipfile import ZipFile
import glob

st.set_page_config(
        page_title="Marine Debris Detection",  # Title of the browser tab
        page_icon=":wave:",  # Favicon (can be an emoji or path to an image)
        layout="wide"  # Optional: Set the layout (either wide or centered)
    )
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
    # Inject CSS into the Streamlit app
    st.markdown("""
    <style>
        /* General Page Styling */
        body {
            background-color: #f0f4f7;
            font-family: 'Arial', sans-serif;
        }

        /* Banner Styling */
        .banner {
            text-align: center;
            font-size: 2em;
            color: #0044cc;
            padding: 20px;
            background-color: #e1efff;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* About Page Content Styling */
        .about-content {
            padding: 20px;
            margin: 20px 0;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .about-content h3 {
            font-size: 1.5em;
            color: #333;
        }

        /* Button Hover Effect */
        .stButton > button:hover {
            background-color: #0056b3;
            color: white;
            transform: scale(1.05);
        }

        /* Removed image animation effect */
        /* 
        .stImage {
            transform: perspective(500px) rotateY(10deg);
            transition: transform 0.5s ease-in-out;
        }

        .stImage:hover {
            transform: perspective(500px) rotateY(0deg);
        }
        */
    </style>
    """, unsafe_allow_html=True)

    st.title("🌊 Marine Debris Semantic Segmentation with U-Net")
    st.markdown('<div class="banner">🚢 Detecting Marine Debris from Satellite Images 🌍</div>', unsafe_allow_html=True)

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

        os.unlink(temp_file_path)
        os.unlink(tiff_path)

def about_page():
    # Inject custom CSS styling
    st.markdown("""
    <style>
    .about-container {
        background-color: #f9f9fc;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
        font-family: 'Segoe UI', sans-serif;
        color: #333;
    }
    .about-container h3 {
        color: #004080;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .about-container ul {
        margin-left: 1.2rem;
    }
    .about-image {
        margin: 1rem 0;
        border-radius: 10px;
        border: 1px solid #ddd;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .about-link {
        color: #0066cc;
        text-decoration: none;
    }
    .about-link:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<div class='about-container'>", unsafe_allow_html=True)
    st.title("📘 About Marine Debris Detection")

    st.markdown("---", unsafe_allow_html=True)

    # Section: What is Marine Debris?
    st.markdown("""
    <h3>🌊 What is Marine Debris?</h3>
    <p>Marine debris refers to human-created waste that ends up in oceans, seas, lakes, and waterways—intentionally or unintentionally.</p>
    <ul>
        <li>Plastics (bags, bottles, microplastics)</li>
        <li>Abandoned fishing gear (ghost nets)</li>
        <li>Rubber, glass, metal, and textiles</li>
    </ul>
    <p>Most marine debris, especially plastic, <strong>does not biodegrade</strong>. Instead, it breaks down into smaller fragments called <strong>microplastics</strong>, which can be ingested by marine animals and even enter the human food chain.</p>
    <img class='about-image' src='https://seahistory.org/wp-content/uploads/marine-debris.jpg' width='100%' />
    <p><em>Image: Marine Debris [Source: <a class='about-link' href='https://seahistory.org/sea-history-for-kids/getting-rid-of-marine-debris/' target='_blank'>National Maritime History Society</a>]</em></p>
    """, unsafe_allow_html=True)

    st.markdown("---", unsafe_allow_html=True)

    # Section: Why It Matters
    st.markdown("""
    <h3>♻️ Why Does It Matter?</h3>
    <ul>
        <li>🐢 <strong>Wildlife Impact:</strong> Animals can become entangled or mistake debris for food.</li>
        <li>🪸 <strong>Habitat Destruction:</strong> Coral reefs and mangroves are especially vulnerable.</li>
        <li>⚓ <strong>Navigation Hazards:</strong> Floating debris can damage boats and ships.</li>
        <li>🧬 <strong>Human Health Risk:</strong> Microplastics have been detected in seafood and even drinking water.</li>
    </ul>
    <img class='about-image' src='https://marine-debris-site-s3fs.s3.us-west-1.amazonaws.com/s3fs-public/sea_turtle_entangled.jpg?VersionId=26WuPYKaZUe82w4GutIHwwK9adTByMaL' width='100%' />
    <p><em>Image: Green sea turtle entangled in fishing line. [Source: <a class='about-link' href='https://marinedebris.noaa.gov/entangled-green-sea-turtle' target='_blank'>NOAA Marine Debris Program</a>]</em></p>
    """, unsafe_allow_html=True)

    st.markdown("---", unsafe_allow_html=True)

    # Section: Role of AI
    st.markdown("""
    <h3>🛰️ Role of Remote Sensing & Deep Learning</h3>
    <p><strong>Remote sensing</strong> using satellites and drones, combined with <strong>semantic segmentation models</strong> like <strong>U-Net++</strong>, can:</p>
    <ul>
        <li>Detect marine debris in multi-spectral satellite images</li>
        <li>Classify pixels into debris types or regions</li>
        <li>Monitor temporal trends in pollution</li>
        <li>Guide clean-up missions and policy planning</li>
    </ul>
    <img class='about-image' src='https://www.esa.int/var/esa/storage/images/esa_multimedia/images/2015/03/sentinel-2/15292661-1-eng-GB/Sentinel-2_pillars.jpg' width='100%' />
    <p><em>Image: Sentinel-2 Satellite. [Source: <a class='about-link' href='https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2' target='_blank'>ESA</a>]</em></p>
    """, unsafe_allow_html=True)

    st.markdown("---", unsafe_allow_html=True)

    # Section: Model Info
    st.markdown("""
    <h3>🧠 About the Model</h3>
    <p>This app uses a custom-trained <strong>U-Net++</strong> model, fine-tuned on multi-band satellite TIFF images.</p>
    <ul>
        <li><strong>Input:</strong> Multi-band (11-channel) imagery</li>
        <li><strong>Output:</strong> Pixel-level classification of debris types</li>
        <li><strong>Framework:</strong> PyTorch</li>
    </ul>
    <img class='about-image' src='https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png' width='100%' />
    <p><em>Image: U-Net Architecture. [Source: <a class='about-link' href='https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/' target='_blank'>University of Freiburg</a>]</em></p>
    """, unsafe_allow_html=True)

    st.markdown("---", unsafe_allow_html=True)

    # Section: Dataset
    st.markdown("""
    <h3>📦 Dataset: MARIDA</h3>
    <p>The model is trained using the <strong>MARIDA (MARine Debris Archive)</strong> dataset:</p>
    <ul>
        <li>Developed by GEOMAR Helmholtz Centre for Ocean Research Kiel</li>
        <li>Classes: Marine debris, ships, clean water, natural organic material (NOM), Sargassum</li>
        <li>Includes polygon annotations and raster masks</li>
        <li>1000+ annotated samples from global coastal areas</li>
    </ul>
    <img class='about-image' src='https://production-media.paperswithcode.com/datasets/1b137f41-d688-438b-9daa-9d3b5d5c3d55.jpg' width='100%' />
    <p><em>Image: MARIDA Dataset. [Source: <a class='about-link' href='https://zenodo.org/record/5151941' target='_blank'>Zenodo</a>]</em></p>
    """, unsafe_allow_html=True)

    st.markdown("---", unsafe_allow_html=True)

    # Section: QGIS
    st.markdown("""
    <h3>🗺️ Visualizing Results with QGIS</h3>
    <p><strong>QGIS</strong> is an open-source Geographic Information System for geospatial analysis.</p>
    <p><strong>Steps to visualize:</strong></p>
    <ol>
        <li>Download QGIS from <a class='about-link' href='https://qgis.org/en/site/' target='_blank'>qgis.org</a></li>
        <li>Load the segmentation_output.tif</li>
        <li>Apply .qml file for color mapping</li>
        <li>Overlay with basemaps or environmental layers</li>
    </ol>
    <img class='about-image' src='https://upload.wikimedia.org/wikipedia/commons/5/5a/QGIS_Interface_Screenshot_with_Map_of_Median_Income_in_Houston_%282010%29.png' width='100%' />
    <p><em>Image: QGIS Interface. [Source: <a class='about-link' href='https://commons.wikimedia.org/wiki/File:QGIS_3.10_Overview.png' target='_blank'>Wikimedia Commons</a>]</em></p>
    """, unsafe_allow_html=True)

    st.markdown("---", unsafe_allow_html=True)

    # Section: Learn More
    st.markdown("""
    <h3>📘 Learn More</h3>
    <ul>
        <li>🌐 <a class='about-link' href='https://marinedebris.noaa.gov/' target='_blank'>NOAA Marine Debris Program</a></li>
        <li>🌐 <a class='about-link' href='https://www.cleanseas.org/' target='_blank'>UNEP Clean Seas Campaign</a></li>
        <li>📊 <a class='about-link' href='https://zenodo.org/record/5151941' target='_blank'>MARIDA Dataset on Zenodo</a></li>
        <li>📄 <a class='about-link' href='https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262247' target='_blank'>MARIDA Paper on PLOS ONE</a></li>
        <li>📚 <a class='about-link' href='https://docs.qgis.org/' target='_blank'>QGIS Documentation</a></li>
        <li>🛰️ <a class='about-link' href='https://www.sentinel-hub.com/' target='_blank'>ESA Sentinel Hub</a></li>
        <li>🧠 <a class='about-link' href='https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/' target='_blank'>U-Net++: Nested U-Net</a></li>
    </ul>
    """, unsafe_allow_html=True)

    # Close the styled container
    st.markdown("</div>", unsafe_allow_html=True)


# 🧭 Navigation
def main_router():
    page = st.sidebar.radio("🔍 Select Page", ["📘 About Marine Debris", "🏠 Segmentation App"])
    
    if page == "📘 About Marine Debris":
        about_page()
    elif page == "🏠 Segmentation App":
        main()

if __name__ == "__main__":
    main_router()
