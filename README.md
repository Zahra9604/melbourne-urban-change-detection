# melbourne-urban-change-detection

Deep learning for building change detection using Sentinel-1/Sentinel-2 imagery and OSCD-to-Melbourne transfer learning.

## Overview

This project investigates deep-learning approaches for detecting building changes in Melbourne's western urban-growth corridor using multi-temporal Sentinel-1 and Sentinel-2 satellite imagery.

The project explores whether **transfer learning from the Onera Satellite Change Detection (OSCD) dataset** can improve change detection when only a limited amount of locally labelled Melbourne data are available.

A particular focus of the project is the interaction between **model predictions and ground-truth development**. Model predictions are used not only for evaluation, but also to identify areas that may require further manual review and ground-truth refinement.

## Study Area

📍 **Melbourne, Victoria, Australia**

The study focuses on Melbourne's western urban-growth corridor, including rapidly developing areas around:

* Werribee
* Tarneit
* Truganina
* Mambourin

The analysis uses a 10 m spatial resolution and focuses on building-related urban change.

## Requirements

### Software

* Python 3.10+
* PyTorch
* NumPy
* Pandas
* Rasterio
* GeoPandas
* Shapely
* scikit-learn
* OpenCV
* Matplotlib
* tqdm
* Google Earth Engine Python API

The complete Python dependencies are listed in `requirements.txt`.

### External Data

Several datasets are required for the complete workflow. The satellite and land-cover data used for the Melbourne study area are **downloaded automatically through the provided Python scripts**, rather than being stored in this repository.

### OSCD Dataset

The **Onera Satellite Change Detection (OSCD)** dataset is required for OSCD pretraining and transfer-learning experiments.

The OSCD dataset must be downloaded separately and is **not included in this repository**.

After downloading and extracting the dataset, the initial structure should be:

```text
OSCD_dataset/
├── Images/
└── Labels/
```

The preprocessing code in `OSCD/Codes/` is then used to prepare the dataset and generate the required training, validation, and test splits.

```text
OSCD Dataset
      ↓
Images + Labels
      ↓
OSCD Preprocessing
      ↓
Train / Validation / Test Split
      ↓
Model Training
      ↓
Transfer Learning on Melbourne
```

### Melbourne Satellite Data

The Melbourne change-detection inputs are generated from **Sentinel-1 SAR** and **Sentinel-2 optical imagery**.

The required Sentinel-1 and Sentinel-2 imagery is downloaded automatically using the **Google Earth Engine Python API** through the provided data-acquisition/preprocessing scripts.

Therefore, users do not need to manually download individual Sentinel-1 and Sentinel-2 scenes.

The workflow is:

```text
Google Earth Engine
        ↓
Sentinel-1 / Sentinel-2
        ↓
AOI and temporal filtering
        ↓
Preprocessing
        ↓
Multi-temporal model inputs
```

Users will need a Google Earth Engine account with access to the Earth Engine Python API and appropriate authentication/configuration for their environment.

### DEA Land Cover

The **Digital Earth Australia (DEA) Land Cover** product is used to generate land-cover-based change candidates that support the development and validation of the Melbourne ground-truth dataset.

DEA Land Cover data are downloaded automatically using:

```text
Melbourne/Codes/DEAmap_download.py
```

The script uses the **Melbourne DEA Land Cover C3 Downloader** to obtain the required DEA Land Cover data for the Melbourne study area.

The workflow is:

```text
DEA Land Cover C3
        ↓
Melbourne DEA Land Cover Downloader
        ↓
DEA Land Cover 2023 / 2025
        ↓
Land-cover change candidates
        ↓
Ground-truth development
```

The DEA Land Cover data are not stored in the public repository.

### High-Resolution Imagery

High-resolution imagery is used for visual validation and manual refinement of potential building changes.

This imagery is used to:

* verify model-generated change candidates
* investigate potential false positives
* identify changes missed by initial labelling
* refine building-change boundaries

## Data

The project uses:

* **Sentinel-1 SAR imagery**
* **Sentinel-2 optical imagery**
* **DEA Land Cover**
* High-resolution imagery for visual validation
* **OSCD** for initial model training and transfer learning

Sentinel-1 and Sentinel-2 data are processed to create multi-temporal inputs for building change detection.

The main Melbourne Sentinel-1/Sentinel-2 input configuration combines:

* Sentinel-1 VV/VH
* Sentinel-2 spectral bands
* Multi-temporal observations

## Ground-Truth Development

An initial Melbourne-specific ground-truth dataset was created by combining:

1. Model-generated change candidates
2. DEA Land Cover-based change candidates
3. Sentinel-2 imagery
4. High-resolution imagery
5. Manual visual interpretation and refinement

The resulting labels represent binary building change:

```text
0 = No change
1 = Building change
```

The ground truth is treated as an **iterative dataset-development process** rather than a completely fixed product.

During model evaluation, predictions from the fine-tuned models were also reviewed to identify potential changes that may have been missed during the initial labelling process.

## Models and Experiments

Several modelling strategies were investigated.

### 1. Early Fusion — OSCD → Melbourne

The multi-modal imagery is combined at the input level. The model is initially trained using OSCD data and subsequently fine-tuned using Melbourne data.

### 2. Dual-Stream — OSCD → Melbourne

Separate processing streams are used for Sentinel-1 and Sentinel-2 information. The model is pretrained on OSCD and then fine-tuned using Melbourne data.

### 3. Early Fusion — Melbourne Only

The Early Fusion model is trained using only the available Melbourne training data.

### 4. Dual-Stream — Melbourne Only

The Dual-Stream model is trained using only the Melbourne dataset.

### 5. OSCD-Only

Models trained on OSCD are directly applied to the Melbourne study area without local fine-tuning.

## Experimental Workflow

```text
OSCD Dataset
      ↓
OSCD Preprocessing
      ↓
Train / Validation / Test
      ↓
OSCD Model Training
      ↓
Melbourne Satellite Data Acquisition
      ↓
Initial Melbourne Prediction
      ↓
Initial Ground-Truth Creation
      ↓
Melbourne Fine-Tuning
      ↓
Model Comparison
      ↓
Identify Potential Missed Changes
      ↓
Manual Imagery Review
      ↓
Ground-Truth Refinement
```

This iterative workflow allows model predictions to contribute to the improvement of the local training dataset.

## Key Findings

The experiments showed that:

* Models trained only on the limited Melbourne dataset were less effective than expected.
* Transfer learning from OSCD provided a useful starting point for the Melbourne study area.
* The **OSCD-pretrained Dual-Stream model fine-tuned on Melbourne data** performed particularly well among the tested approaches.
* Model predictions highlighted several apparent building changes that were not included in the initial ground-truth labels.
* Manual review of these locations indicated that some changes may have been missed during the initial labelling process.

These observations suggest that transfer learning can be useful not only for model development, but also for **AI-assisted ground-truth refinement when local labelled data are limited**.

## Repository Structure

```text
melbourne-urban-change-detection/
│
├── Melbourne/
│   ├── Codes/
│   └── MelbourneData/
│
├── OSCD/
│   ├── Codes/
│   └── Data/
│
├── README.md
└── requirements.txt
```

## Technologies

* GIS / Remote Sensing
* Google Earth Engine
* Python
* PyTorch
* Sentinel-1
* Sentinel-2
* Digital Earth Australia (DEA)
* Deep Learning
* Change Detection
* Transfer Learning
* Spatial Data Processing

## Author

**Tia Azarm**

Geospatial AI Specialist | GIS | Remote Sensing | Spatial Analytics
