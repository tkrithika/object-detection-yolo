# object-detection-yolo
Training custom object detection models using YOLO in Google Colab

# YOLO Object Detection Training using Google Colab

## Overview

This repository demonstrates a complete workflow for training a custom object detection model using YOLO and Google Colab. The project covers dataset collection, annotation, model training on a free GPU, and running inference on images, videos, or a USB camera.

The example use case focuses on **pothole detection**, but the same workflow can be adapted for any object detection task.

---

## Tools & Technologies

* **Google Colab** – Cloud-based Python environment with free GPU support
* **YOLO (v5 / v8 / v11)** – Object detection framework
* **Label Studio** – Data annotation tool
* **Kaggle** – Dataset source
* **Roboflow** – Dataset management and conversion
* **Open Images V7** – Public image dataset
* **Anaconda** – Python environment and package manager

---

## Useful Resources

* Label Studio: [https://labelstud.io/](https://labelstud.io/)
* Label Studio Documentation (Quick Start): [https://labelstud.io/guide/](https://labelstud.io/guide/)
* Roboflow: [https://roboflow.com/](https://roboflow.com/)
* Open Images V7 Dataset: [https://storage.googleapis.com/openimages/web/index.html](https://storage.googleapis.com/openimages/web/index.html)
* Kaggle Datasets: [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
* YOLO Reference Video: [https://www.youtube.com/watch?v=r0RspiLG260&t=208s](https://www.youtube.com/watch?v=r0RspiLG260&t=208s)

---

## Dataset Preparation

### Dataset Source

* Download a pothole object detection dataset from **Kaggle**, **Roflowflow**, or **open images V7 dataset**
* Save the dataset as a ZIP file, inside documents in your PC under a folder named "yolo". 

### Why Label Studio?

Label Studio is a data annotation tool used to create labeled datasets for training custom AI models.

**What Label Studio does:**

* Allows uploading images, videos, text, or audio
* Supports bounding boxes, polygons, and keypoints
* Lets you define custom labels (e.g., pothole)
* Exports annotations in YOLO, COCO, JSON, and CSV formats

**Why it is important:**

* Object detection models require accurately labeled data
* Without annotations, a model cannot learn
* Label Studio provides a clean GUI to create professional datasets

> If you are using only pre-trained YOLO models, Label Studio is not required.

---

## Annotation Workflow

1. Create a new project in Label Studio
2. Choose **Object Detection with Bounding Boxes**
3. Remove default labels and add custom labels (e.g., `pothole`)
4. Import images into the project
5. Draw bounding boxes around potholes
6. Export annotations in **YOLO with images** format

---

## Model Training (Google Colab)

Google Colab is used for training because it provides free GPU access and requires no local setup.

## Open in Google Colab

Click the link below to open the training notebook in Google Colab.
You must be logged into your Google account to run the notebook.


https://colab.research.google.com/github/tkrithika/object-detection-yolo/blob/main/Train_YOLO_Models.ipynb



### Steps:

1. Open the training notebook in Google Colab
2. Set Runtime Type to **GPU (Tesla T4)**
3. Upload the exported dataset ZIP file (`data.zip`) via the Colab file panel
4. Run all cells in the notebook
5. After training, download the generated `my_model.zip`

---

## Training Notebook

* **Notebook Name:** `Train_YOLO_Models.ipynb`
* This notebook handles:

  * Environment setup
  * Dataset extraction
  * Model configuration
  * Training
  * Validation

---

## Inference (Testing the Trained Model)

### What is Inference?

Inference refers to running a trained YOLO model on new images or videos to detect objects.

During inference, the model outputs:

* Bounding boxes
* Class labels
* Confidence scores

---

## Anaconda Environment Setup (Optional)

Anaconda helps manage Python environments and avoid dependency conflicts.

### Basic Setup

```bash
conda create --name yolo-env1 python=3.12
conda activate yolo-env1
```

After this you will see (yolo-env1) 
Get inside the path where your "my_model.pt" is present

```bash
pip install ultralytics
```

Next in your browser search for:
http://pytorch.org/get-started/locally/ 
<img width="975" height="325" alt="image" src="https://github.com/user-attachments/assets/d355029e-dbff-45d6-8f43-0de86d65d854" />

Install PyTorch (CUDA-enabled) inisde Anaconda prompt:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Running Inference Locally
After this Create a new folder named "yolo_detect" (python script) and place it under the same folder as my_model.pt
The `yolo_detect.py` script is used for running inference on images, videos, or a USB camera.

---
### Example Commands

**USB Camera:**

```bash
python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720
```

**Video File:**

```bash
python yolo_detect.py --model my_model.pt --source sample_video.mp4
```

---
## Project Output

* Trained YOLO model weights
* Detection results on images or videos
* Real-time detection using a USB camera

---

## Notes

* Large datasets and trained models are not included in this repository
* Users should upload their own datasets to Google Drive or Colab
* This repository is intended for learning and experimentation

---

## Author

Created as a beginner-friendly guide for training YOLO object detection models using Google Colab.
