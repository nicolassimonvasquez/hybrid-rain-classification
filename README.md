# Hybrid Rain Classification System

This repository contains the source code for a deep learning system designed to classify rain intensity (Clear, Light, Heavy) using a hybrid R3D18 and MLP architecture.

## Project Structure
* **models/**: Contains the R3D18 backbone architecture.
* **utils/**: Helper functions for data processing.
* **final-feature-extraction.py**: Extracts features using R3D18 and DCP.
* **final-train-mlp-v2.py**: Trains the MLP classifier on extracted features.
* **final-real-time-test.py**: Real-time inference script using OpenCV.

## Training Dataset Statistics (Sequential 80/20 Split)
The system was trained strictly on frames extracted from Youtube videos.
| Class | Train | Val | Total |
| :--- | :--- | :--- | :--- |
| Clear | 5998 | 1500 | 7498 |
| Light Rain | 5996 | 1500 | 7496 |
| Heavy Rain | 5996 | 1500 | 7496 |
| **TOTAL** | **17988** | **4500** | **22488** |

## Getting Started
1. Clone the repository: `git clone https://github.com/nicolassimonvasquez/hybrid-rain-classification.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the trained weights (see link below).

## Model Weights
Due to file size limitations, the `.pth` checkpoints are hosted externally:
* https://drive.google.com/drive/folders/1SzcCFFYSTnx6CBS-fmDNfHKjazAu6cyR?usp=drive_link
  * **mlp-best-rigorous-weights** - `rigorous_epoch_18.pth`
  * **mlp_best_checkpoint_rigorous** - `checkpoint_epoch_4.pth`

## System Components
1. Feature Extraction (`final-feature-extraction.py`)
    * **Purpose:** Extracts high-dimensional features from raw video frames to prepare a dataset for the MLP classifier.
    * **Deep Features:** Uses an R3D-18 (3D ResNet) backbone to capture temporal motion and spatial details, producing a 512-dimensional feature vector.
    * **Physics Feature (DCP):** Implements Dark Channel Prior logic to calculate a "haze score," providing an objective measure of atmospheric visibility.   
    * **Data Augmentation:** Includes logic to apply random brightness and noise artifacts to 30% of sequences to improve model robustness.

2. MLP Training (`final-train-mlp-v2.py`)
    * **Purpose:** Trains a Multi-Layer Perceptron (MLP) to classify weather intensity based on the hybrid features.
    * **Architecture:** A 4-layer fully connected network (513 input features $\rightarrow$ 256 $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 3 classes) with Dropout layers to prevent overfitting.
    * **Rigorous Split:** Implements a sequential 80/20 split (rather than random shuffling) to ensure the validation set represents unseen future time-steps, which is critical for time-series meteorological data.

3. Real-Time Inference Testing (`final-real-time-test.py`)
    * **Purpose:** Deploys the trained hybrid model for live monitoring testing.
    * **Sliding Window:** Processes video using a 16-frame window with a stride of 4, ensuring smooth transitions in the classification output.
    * **Live Support:** Features a specialized version for RTSP live streams (e.g., Tapo cameras) with a reduced buffer size to minimize latency during real-time weather monitoring. You must be connected to AIC's dedicated wi-fi to access Tapo cameras.

## Authorship & Affiliation
This project was developed by **Nicolas Simon Vasquez** during an internship at the **Ateneo Innovation Center (AIC)**. 
* **Developer:** Nicolas Simon Vasquez
* **Supervising Organization:** Ateneo Innovation Center (AIC)
* **Project Context:** Part of an ongoing initiative for real-time disaster and risk monitoring management.

All rights and code logic are shared between the author and AIC. For access to the dedicated AIC Wi-Fi for RTSP camera testing, please contact the laboratory administrator.
