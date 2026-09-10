# melbourne-urban-change-detection

Deep learning for building change detection using Sentinel-1/Sentinel-2 imagery and OSCD-to-Melbourne transfer learning.

## Overview

This project investigates deep-learning approaches for detecting building changes in Melbourne's western urban-growth corridor using multi-temporal Sentinel-1 and Sentinel-2 satellite imagery.

The project explores whether **transfer learning from the Open Sentinel-2 Change Detection (OSCD) dataset** can improve change detection when only a limited amount of locally labelled Melbourne data are available.

A particular focus of the project is the interaction between **model predictions and ground-truth development**. Model predictions are used not only for evaluation, but also to identify areas that may require further manual review and ground-truth refinement.

## Study Area

📍 **Melbourne, Victoria, Australia**

The study focuses on Melbourne's western urban-growth corridor, including rapidly developing areas around:

* Werribee
* Tarneit
* Truganina
* Mambourin

The analysis uses a 10 m spatial resolution and focuses on building-related urban change.

## Data

The project uses:

* **Sentinel-1 SAR imagery**
* **Sentinel-2 optical imagery**
* **DEA Land Cover** products
* High-resolution imagery for visual validation
* **OSCD** for initial model training

The Sentinel-1 and Sentinel-2 data are processed to create multi-temporal inputs for change detection.

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

Several modelling strategies were investigated:

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
OSCD Training
      ↓
Melbourne Prediction
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
├── README.md
├── data/
├── models/
├── preprocessing/
├── training/
├── inference/
├── evaluation/
├── notebooks/
├── results/
├── requirements.txt
└── .gitignore
```

## Status

🚧 **Work in progress**

The Melbourne ground-truth dataset is currently being systematically reviewed and refined. Additional experiments, evaluation, and visualisations will be added as the project develops.

## Future Work

Planned improvements include:

* Further refinement of Melbourne ground-truth labels
* Additional model evaluation
* More systematic comparison of fusion strategies
* Quantitative evaluation on the refined dataset
* Improved visualisation of predictions and change maps
* Further investigation of transfer learning for limited-label geospatial applications

## Technologies

* Python
* PyTorch
* Google Earth Engine
* Sentinel-1
* Sentinel-2
* GIS / Remote Sensing
* Deep Learning
* Change Detection
* Transfer Learning

## Author

**Tia Azarm**

Geospatial AI Specialist | GIS | Remote Sensing | Spatial Analytics

