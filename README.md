# 🌊 Marine Debris Detection using UNet++

A deep learning-based solution for detecting floating marine debris from satellite imagery. This project leverages the **UNet++** architecture and the **MARIDA dataset**, along with custom loss functions, preprocessing, and augmentation techniques to improve segmentation accuracy under diverse oceanic conditions.

---

## 🚀 Deployed Web Application

👉 **[Click here to launch the demo app](https://aqua-scan.streamlit.app)**  
> Upload a Sentinel-2 image and get an instant prediction of marine debris segments.

---

## 📌 Features

- 🔍 Semantic segmentation using UNet++ (Nested U-Net)
- 📦 MARIDA dataset support with preprocessing and augmentation
- 🎯 Custom loss functions (Dice + Focal)
- 📈 Metrics: IoU, Dice Score, F1-Score, Accuracy
- 🧪 Real-time prediction via Streamlit web interface
- ☁️ Cloud & sun-glint masking + spectral band normalization
- 🧠 Deep supervision & dense skip connections

---

## 🧠 Architecture Overview: UNet++

UNet++ builds upon traditional U-Net by:

- Adding **nested dense skip connections** for enhanced feature propagation
- Introducing **deep supervision** to improve gradient flow
- Achieving better segmentation performance with **limited annotated data**

🔗 Paper: [UNet++: Redesigning Skip Connections](https://arxiv.org/abs/1912.05074)

---

## 🗂️ Project Structure
   
marine-debris-detection/
├── assets.py # Static paths and constants
├── dataloader.py # Data augmentation and loading pipeline
├── evaluation.py # Prediction, post-processing, visualization
├── loss_function.py # Custom loss (Dice, Focal, Combo)
├── metrics.py # Evaluation metrics (IoU, Dice, F1)
├── unet_plus_plus.py # UNet++ model definition
├── train.py # Model training logic
├── app.py # Streamlit web app interface
├── requirements.txt # Python dependencies
└── README.md # This file

---

## 📊 Dataset

- 📘 **Primary:** [MARIDA Dataset](https://doi.org/10.5281/zenodo.5151941) (Sentinel-2 imagery with polygon labels)
- 📂 Additional: Synthetic augmentations, TrashCan, SeaClear, Sonar datasets

**Preprocessing Techniques:**

- Spectral band alignment & normalization  
- Sun-glint/cloud masking  
- Random flips, rotations, hue/contrast changes  
- Class balancing for debris vs water  

---

## 💻 Local Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/marine-debris-detection.git
   cd marine-debris-detection
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

3. **(Optional) Train the Model**
   ```bash
   python train.py

4. **Run the Streamlit Web App**
   ```bash
   streamlit run app.py

## 📈 Evaluation Metrics
- Implemented in metrics.py:
- Intersection over Union (IoU)
- Dice Coefficient
- Precision, Recall, F1 Score
- Per-class Accuracy


## 🧪 Code Snippet Overview
**File	Functionality Summary**
- assets.py	Static config: band names, paths, colors
- metrics.py	IoU, Dice, F1-score, accuracy calculations
- dataloader.py	Loads & augments MARIDA images/masks
- unet_plus_plus.py	Defines the UNet++ architecture in PyTorch
- loss_function.py	Custom losses for segmentation training
- evaluation.py	Evaluates model output & renders masks
- train.py	Trains model with logs & checkpointing
- app.py	Streamlit-based interface for live testing

## 📚 Key References
- Kikaki et al. (2020) – MARIDA Dataset Link
- Zhou et al. (2019) – UNet++ Architecture Link
- Mohammed (2022) – ResAttUNet
- Gupta et al. (2023) – MFPN Network
- Zocco et al. (2022) – EfficientDets for real-time debris detection
- Zhou et al. (2022) – YOLOTrashCan

## 🙌 Acknowledgements
- MARIDA Team
- ESA Copernicus Sentinel-2 Mission
- OpenAI for technical guidance and documentation support
