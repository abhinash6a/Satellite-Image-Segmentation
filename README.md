# 🛰️ Satellite Image Segmentation

An **AI-powered Land Use Classification System** that performs **semantic segmentation** on satellite imagery.  
This project identifies different land cover types such as **Buildings**, **Roads**, **Vegetation**, and **Water Bodies** using a deep learning segmentation model.

---
<img width="1480" height="1306" alt="image" src="https://github.com/user-attachments/assets/e24b29cb-ded2-4def-b226-3b2007805275" />


## 🚀 Features

- 🌍 Upload and process high-resolution satellite images (PNG, JPG, TIF)
- 🧠 Deep Learning-based land use classification
- 🖼️ Visualize:
  - Original Image
  - Segmentation Mask
  - Overlay (Segmentation + Original)
- 📊 Class Distribution summary for each image
- 📥 Downloadable segmentation masks and overlays
- 💡 Clean and interactive web UI built for intuitive exploration

---

## 🧩 Segmentation Classes

| Class Name   | Color  | Description |
|---------------|---------|-------------|
| **Background** | ⚫ Black | Non-relevant or empty regions |
| **Buildings**  | 🔴 Red   | Man-made constructions |
| **Roads**      | ⚪ Grey  | Streets, highways, and roads |
| **Vegetation** | 🟢 Green | Trees, plants, and green cover |
| **Water**      | 🔵 Blue  | Rivers, lakes, and water bodies |

---


## 🧠 Tech Stack

- **Frontend:** React.js, TailwindCSS, ShadCN UI  
- **Backend:** Flask / FastAPI (Python)  
- **Model:** UNet / DeepLabV3+ / SegFormer (via PyTorch or TensorFlow)  
- **Libraries:**  
  - `segmentation_models_pytorch`  
  - `albumentations`  
  - `OpenCV`  
  - `torch` / `tensorflow`  
  - `PIL`, `NumPy`, `tqdm`
