🧬 Protein Pre-Cancer Prediction Using CNN

An end-to-end AI system for predicting cancer-associated proteins using AlphaFold-generated 3D structures, RGB biophysical image encoding, ensemble CNN models, and Grad-CAM explainability.

    📌 Table of Contents
    
    Overview
    
    Motivation
    
    System Architecture
    
    Methodology
    
    Preprocessing Pipeline
    
    Model Architecture
    
    Training Strategy
    
    Visualization & Explainability
    
    Datasets Used
    
    Results
    
    Project Structure
    
    How to Run
    
    Requirements
    
    Limitations
    
    Future Enhancements
    
    License
    
    Acknowledgements

🔍 Overview

    This project presents a deep learning framework to predict whether a protein is cancer-associated based solely on its 3D structural information.
    
    Instead of relying on protein sequences or handcrafted biological features, the system:
    
    Converts protein structures into RGB biophysical images
    
    Uses an ensemble of convolutional neural networks
    
    Provides explainability using Grad-CAM

🎯 Motivation

    Most traditional cancer protein prediction systems rely on:
    
    Sequence data
    
    Omics features
    
    They often ignore 3D structural alterations caused by mutations.
    
    With the availability of high-quality protein structures from AlphaFold, this project exploits structural cues using modern deep learning techniques.

🏗 System Architecture

    The system consists of the following modules:
    
    Data Integration – OncoKB, UniProt, AlphaFold
    
    Preprocessing Engine – 3D → 2D biophysical encoding
    
    Dataset Management – Cancer vs Non-Cancer labeling
    
    Training Core – CNN ensemble training
    
    Visualization Layer – Grad-CAM + 3D protein viewer
    
    Deployment Layer – Streamlit web application

⚙ Methodology

    Collect cancer gene information from OncoKB
    
    Map genes to UniProt identifiers
    
    Download protein 3D structures from AlphaFold
    
    Extract Cα atoms and confidence scores from .pdb files
    
    Generate:
    
    Distance matrices
    
    Stability maps
    
    Depth maps
    
    Encode biophysical features into 299×299 RGB images
    
    Train an ensemble CNN model
    
    Perform inference with Grad-CAM explainability

🧪 Preprocessing Pipeline

    Cα atom extraction
    
    Pairwise distance matrix computation
    
    B-factor (confidence score) extraction
    
    Depth map calculation
    
    RGB channel encoding
    
    🔴 Distance
    
    🟢 Confidence
    
    🔵 Depth
    
    Padding and resizing to CNN-compatible format

🧠 Model Architecture

    The system uses an ensemble of three pretrained CNN models:
    
    DenseNet201 – Deep feature reuse
    
    EfficientNet-B4 – Lightweight and efficient
    
    SE-ResNet50 – Channel attention mechanism
    
    Final prediction is obtained using ensemble averaging.

🏋 Training Strategy

    Train–Test Split: 80% / 20%
    
    Loss Function: Focal Loss
    
    Optimizer: Adam / AdamW
    
    Learning Rate Scheduler: Cosine Annealing
    
    Data Augmentation:
    
    Flips
    
    Rotations
    
    Normalization
    
    Class imbalance handled using weighted sampling

🔍 Visualization & Explainability

    Grad-CAM heatmaps highlight structurally important regions
    
    3D protein viewer for interactive inspection
    
    Model-wise probability distribution output

🧬 Datasets Used

    OncoKB – Cancer gene annotations
    
    UniProt – Protein metadata
    
    AlphaFold Protein Structure Database – 3D protein structures

📊 Results

    Accuracy: > 98%
    
    Recall (Cancer class): > 85%
    
    ROC-AUC: 0.97
    
    DenseNet201 and SE-ResNet50 showed the strongest performance.

    📁 Project Structure
    Protein_Pre_Cancer_Prediction/
    │
    ├── preprocessing/
    ├── training/
    ├── models/
    ├── app/
    ├── data/
    ├── results/
    ├── README.md
    ├── requirements.txt
    └── LICENSE
▶ How to Run
pip install -r requirements.txt
streamlit run app/app.py

📦 Requirements

    Python 3.10+
    CUDA 11.8
    PyTorch 2.7.1+cu11
    torchvision
    timm
    numpy
    opencv-python
    matplotlib
    streamlit
    pytorch-grad-cam
    py3Dmol

⚠ Limitations

    Binary classification only (Cancer vs Non-Cancer)
    
    Requires .pdb structure as input
    
    Does not classify specific cancer types

🚀 Future Enhancements

    Multi-class cancer type classification
    
    Sequence + structure hybrid models
    
    Mutation-level structural analysis
    
    Transformer-based protein models
    
    Clinical decision-support integration

📜 License

This project is licensed under the MIT License.

🙏 Acknowledgements

AlphaFold Protein Structure Database

OncoKB

UniProt

PyTorch & timm community
