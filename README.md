# 🛰️ Melbourne Urban Change Detection

Deep learning for urban building-change detection using Sentinel-1 SAR, Sentinel-2 multispectral imagery, transfer learning, and Melbourne-specific ground-truth development.

This project investigates how deep-learning change-detection models trained on the **Open Sentinel-2 Change Detection (OSCD)** dataset can be transferred to a different geographic environment: **Melbourne's western urban-growth corridor, Victoria, Australia**.

The project compares **six modelling configurations** using Early Fusion and Dual-Stream architectures, with and without OSCD pretraining.

The main finding is that the **Dual-Stream model pretrained on OSCD and fine-tuned on Melbourne achieved the best performance on the Melbourne dataset**.

---
## Practical Result

The final system was applied to Melbourne's western urban-growth corridor
to produce a 10 m building-change map.

The workflow combines Sentinel-1 SAR, Sentinel-2 multispectral imagery,
DEA Land Cover data, deep-learning predictions, and manually refined
training labels to identify areas of potential building development.

![Melbourne Building Change Detection Result](Melbourne/MelbourneData/results/test_model_comparison)

The resulting map provides a spatially explicit view of where building
changes were detected between the selected observation periods. This
demonstrates how the developed models can be used as a practical
geospatial monitoring tool rather than only as a model-comparison exercise.

## 🎯 Project Objectives

The project aims to:

1. Develop a multimodal change-detection workflow using Sentinel-1 and Sentinel-2 imagery.
2. Train Early Fusion and Dual-Stream models on the OSCD dataset.
3. Develop a Melbourne-specific 10 m building-change dataset.
4. Investigate whether OSCD pretraining improves change detection in Melbourne.
5. Compare transfer learning against training from scratch.
6. Compare Early Fusion and Dual-Stream architectures.
7. Investigate how model predictions can assist in identifying potentially missed changes during ground-truth development.

---

# 📍 Study Area

The Melbourne study area covers part of Melbourne's western urban-growth corridor, including areas around:

* Werribee
* Tarneit
* Truganina
* Mambourin

The project focuses on **building-related urban change** between two observation periods.

The Melbourne dataset is processed on a **10 m Sentinel-2 spatial grid**.

---

# 🔬 Overall Workflow

The project consists of two major stages:

### 1. OSCD model development

Models are trained on the OSCD dataset using paired Sentinel-1 and Sentinel-2 observations.

### 2. Melbourne transfer learning and model comparison

The trained OSCD models are transferred to Melbourne and compared against models trained only on Melbourne data.

```text
                 OSCD Dataset
                      │
          ┌───────────┴───────────┐
          │                       │
     Sentinel-1              Sentinel-2
          │                       │
          └───────────┬───────────┘
                      ▼
             OSCD Model Training
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
     Early Fusion            Dual Stream
          │                       │
          └───────────┬───────────┘
                      ▼
              OSCD pretrained
                  models
                      │
                      ▼
              Melbourne dataset
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   OSCD → Melbourne          Melbourne only
     fine-tuning               training
          │                       │
          └───────────┬───────────┘
                      ▼
               Model comparison
                      │
                      ▼
          Best Melbourne model:
        OSCD → Melbourne Dual Stream
```

---

# 🛰️ Data Sources

## Open Sentinel-2 Change Detection (OSCD)

The OSCD dataset is used as the source dataset for model pretraining.

The temporal imagery is processed as:

```text
Sentinel-1:
T1 + T2 → 4 channels

Sentinel-2:
T1 + T2 → 8 channels
```

OSCD labels are converted to binary change labels:

```text
0 = No change
1 = Change
```

---

## Sentinel-1

Sentinel-1 SAR imagery provides complementary information to optical Sentinel-2 imagery.

The two temporal observations are stacked:

```text
S1 T1 + S1 T2 → 4-channel input
```

---

## Sentinel-2

Sentinel-2 multispectral imagery provides optical information for detecting changes between the two observation periods.

```text
S2 T1 + S2 T2 → 8-channel input
```

---

## DEA Land Cover C3

Digital Earth Australia (DEA) Land Cover C3 is used to generate candidate urban-change areas for the Melbourne study area.

The DEA product is originally 30 m resolution and is resampled to the Sentinel-2 10 m grid.

Candidate urban change is identified using:

```text
2023 ≠ Artificial Surface
AND
2025 = Artificial Surface
```

The candidate map contains:

```text
0   = No candidate urban change
1   = Candidate urban change
255 = NoData
```

DEA-derived candidates are **not used directly as final ground truth**. They are used to help identify areas requiring manual inspection.

---

# ⚙️ OSCD Data Preparation

## 1. Dataset splitting

`split_OSCD.py` creates the train, validation, and test directory structure and prepares the corresponding OSCD labels.

```text
OSCD/Data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

The image directories are initially empty and are populated by the Sentinel-1/Sentinel-2 download process.

---

## 2. Sentinel-1/Sentinel-2 acquisition

`download_oscd_s1s2.py` downloads the Sentinel imagery required for the OSCD train, validation, and test datasets.

---

## 3. Label georeferencing

Some OSCD labels do not contain valid CRS or geotransform information.

`preprocess.py` creates georeferenced label GeoTIFFs using the corresponding Sentinel-2 spatial reference when the label dimensions match the Sentinel-2 grid.

```text
OSCD/Data/labels_crs_matched/
├── train/
├── val/
└── test/
```

---

## 4. Patch creation

`create_patches.py` reads:

```text
S1 T1
S1 T2
S2 T1
S2 T2
Label
```

and creates 256 × 256 patches.

The inputs are constructed as:

```text
S1 T1 + S1 T2 → 4 channels
S2 T1 + S2 T2 → 8 channels
```

The resulting dataset is:

```text
OSCD/Data/patches/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Example:

```text
brasilia_0_s1.tif
brasilia_0_s2.tif
brasilia_0_cm.tif
```

---

## 5. CSV generation

`create_csv.py` creates:

```text
train.csv
val.csv
test.csv
```

containing the paths to the corresponding Sentinel-1, Sentinel-2, and label patches.

---

# 🤖 Model Architectures

Two architectures are investigated.

## Early Fusion

Sentinel-1 and Sentinel-2 information are combined before the main feature-extraction network.

```text
S1 T1/T2 ──┐
           ├──► Fusion ──► Network ──► Change Map
S2 T1/T2 ──┘
```

## Dual Stream

Sentinel-1 and Sentinel-2 are processed using separate streams before their learned features are combined.

```text
             ┌──► S1 Stream ──┐
S1 T1/T2 ────┤                │
             │                ├──► Fusion ──► Change Map
S2 T1/T2 ────┤                │
             └──► S2 Stream ──┘
```

The Dual-Stream architecture preserves the separate SAR and optical representations before fusion.

---

# 🧪 Six Model Experiments

The main experiment compares **six configurations**.

| Experiment | Architecture | Training data    | OSCD pretraining |
| ---------- | ------------ | ---------------- | ---------------- |
| 1          | Early Fusion | OSCD             | No               |
| 2          | Early Fusion | OSCD + Melbourne | Yes              |
| 3          | Dual Stream  | OSCD             | No               |
| 4          | Dual Stream  | OSCD + Melbourne | Yes              |
| 5          | Early Fusion | Melbourne only   | No               |
| 6          | Dual Stream  | Melbourne only   | No               |

The experiments allow two main questions to be investigated:

### Does transfer learning help?

```text
OSCD
  ↓
Pretrained model
  ↓
Melbourne fine-tuning
```

compared with:

```text
Melbourne
  ↓
Random initialization
```

### Does architecture matter?

```text
Early Fusion
       vs.
Dual Stream
```

---

# 🔄 Transfer Learning

For the transfer-learning experiments, the model is first trained on OSCD and the learned model weights are then used to initialize the Melbourne model.

```text
OSCD
 │
 ▼
Train model
 │
 ▼
OSCD pretrained weights
 │
 ▼
Melbourne dataset
 │
 ▼
Fine-tune
 │
 ▼
Melbourne model
```

The Melbourne training process uses the OSCD model weights as initialization rather than treating the OSCD model as the final Melbourne model.

---

# 📍 Melbourne Dataset Development

A Melbourne-specific 10 m building-change dataset was developed before the final Melbourne model comparison.

The workflow combines:

* OSCD-trained model predictions
* DEA Land Cover change candidates
* Sentinel-1 imagery
* Sentinel-2 imagery
* high-resolution imagery
* manual visual interpretation

```text
OSCD-trained model
        │
        ▼
Melbourne prediction
        │
        ├──────────────┐
        ▼              ▼
DEA candidates    Satellite imagery
        │              │
        └───────┬──────┘
                ▼
          Manual review
                │
                ▼
       Melbourne 10 m labels
```

The final labels are binary:

```text
0 = No change
1 = Building change
```

and are stored in:

```text
Melbourne/MelbourneData/change_labels_10m/
```

---

# ✏️ Manual Label Refinement

The OSCD-trained model predictions are **not treated as ground truth**.

Instead, they are used to identify locations that may contain building changes.

Using `changemaps_gui.py`, candidate areas are manually reviewed against the available imagery.

During this process:

* missed building changes can be added;
* false positives can be removed;
* uncertain areas can be inspected;
* change boundaries can be refined.

This produces a Melbourne-specific ground-truth dataset rather than simply using automated predictions as labels.

---

# 🧠 Melbourne Training

After the Melbourne ground-truth dataset is created, the six model configurations are trained and evaluated.

The transfer-learning experiments use:

```text
OSCD pretrained model
        ↓
Melbourne training data
        ↓
Fine-tuning
```

while the Melbourne-only experiments use random initialization:

```text
Melbourne training data
        ↓
Random initialization
        ↓
Training
```

Model checkpoints are stored under:

```text
Melbourne/MelbourneData/checkpoints/
```

and experiment results are stored under:

```text
Melbourne/MelbourneData/results/
```

---

# 📊 Model Comparison

The six trained models are compared on the Melbourne dataset.

The comparison evaluates the effect of:

* architecture;
* OSCD pretraining;
* Melbourne fine-tuning;
* training only with local data.

### Main result

The **Dual-Stream model pretrained on OSCD and fine-tuned on Melbourne produced the best overall results on the Melbourne dataset** among the six tested configurations.

```text
                         Melbourne Performance
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              Early Fusion                Dual Stream
                    │                           │
          ┌─────────┴─────────┐       ┌────────┴─────────┐
          │                   │       │                  │
       OSCD only         OSCD→ML    OSCD only        OSCD→ML
          │                   │       │                  │
          │                   │       │                  ▼
          │                   │       │             BEST RESULT
          │                   │       │
          └──────────┬────────┘       └─────────┬────────┘
                     │                          │
                  ML only                    ML only
```

The result suggests that combining **OSCD pretraining, Melbourne fine-tuning, and separate Sentinel-1/Sentinel-2 feature streams** was more effective for this Melbourne study area than the other tested configurations.

---

# 🔎 Model-Assisted Ground-Truth Refinement

An additional observation emerged during the Melbourne experiments.

The OSCD-pretrained and Melbourne-fine-tuned Dual-Stream model identified several apparent building changes that were not present in the initial Melbourne ground-truth labels.

These locations were manually revisited using the original imagery.

This led to an iterative dataset-development workflow:

```text
Initial Melbourne labels
          ↓
Model training
          ↓
Melbourne prediction
          ↓
Potential missed changes
          ↓
Manual imagery review
          ↓
Ground-truth refinement
```

Therefore, the project explores not only **model transfer learning**, but also the potential use of model predictions as a tool for improving locally generated training data.

Importantly, candidate predictions are manually reviewed before being incorporated into the labels.

---

# 🗺️ Visual Validation

QGIS is used to inspect and compare:

* Sentinel-1 imagery
* Sentinel-2 imagery
* DEA Land Cover
* model predictions
* candidate change maps
* Melbourne ground-truth labels
* final model outputs

Visual inspection complements numerical evaluation by allowing individual change detections, false positives, missed changes, and spatial boundaries to be examined.

---

# 📁 Repository Structure

```text
melbourne-urban-change-detection/
│
├── README.md
├── Requirements.txt
│
├── OSCD/
│   ├── Codes/
│   │   ├── create_csv.py
│   │   ├── create_patches.py
│   │   ├── download_oscd_s1s2.py
│   │   ├── models.py
│   │   ├── oscd_dataset.py
│   │   ├── predict_change_maps_oscd.py
│   │   ├── preprocess.py
│   │   ├── split_OSCD.py
│   │   ├── test.py
│   │   ├── train_oscd.py
│   │   ├── training_config.py
│   │   └── utils.py
│   │
│   ├── Data/
│   └── logs/
│
└── Melbourne/
    ├── AOI_shp/
    │
    ├── Codes/
    │   ├── changemaps_gui.py
    │   ├── compare_models.py
    │   ├── create_csv.py
    │   ├── DEAmap_download.py
    │   ├── download_melbourne_s1s2.py
    │   ├── melbourne_dataset.py
    │   ├── models.py
    │   ├── prediction_maps.py
    │   ├── preprocess_ML.py
    │   ├── shapefile_AOI.py
    │   ├── train.py
    │   └── training_config.py
    │
    └── MelbourneData/
        ├── change_labels_10m/
        ├── change_maps/
        ├── checkpoints/
        └── results/
```

---

# 🛠️ Technologies

### Deep Learning

* Python
* PyTorch
* Convolutional Neural Networks
* Early Fusion
* Dual-Stream networks
* Transfer learning
* Mixed-precision training

### Remote Sensing

* Sentinel-1 SAR
* Sentinel-2 multispectral imagery
* DEA Land Cover C3
* Google Earth Engine

### Geospatial

* QGIS
* ArcGIS Pro
* Rasterio
* GeoPandas
* NumPy
* GeoTIFF
* CRS and raster reprojection/resampling

---

# 🚀 Reproducibility

The complete workflow can be summarized as:

```text
OSCD
 │
 ├── Split dataset
 ├── Download Sentinel-1/Sentinel-2
 ├── Match label georeferencing
 ├── Create 256 × 256 patches
 ├── Create CSV files
 │
 ▼
Train Early Fusion + Dual Stream models on OSCD
 │
 ▼
Evaluate OSCD models
 │
 ▼
Generate OSCD change maps
 │
 ▼
Melbourne AOI
 │
 ├── Download Sentinel-1/Sentinel-2
 ├── Download DEA Land Cover
 ├── Generate DEA candidate maps
 │
 ▼
Apply OSCD models to Melbourne
 │
 ▼
Manually create/refine Melbourne 10 m labels
 │
 ▼
Train six Melbourne model configurations
 │
 ├── Early Fusion — OSCD only
 ├── Early Fusion — OSCD → Melbourne
 ├── Dual Stream — OSCD only
 ├── Dual Stream — OSCD → Melbourne
 ├── Early Fusion — Melbourne only
 └── Dual Stream — Melbourne only
 │
 ▼
Compare Melbourne results
 │
 ▼
Best performing configuration:
Dual Stream — OSCD → Melbourne
```

---

# 🌏 Key Finding

The main result of this project is that **transfer learning from OSCD to Melbourne improved the performance of the Dual-Stream architecture**, with the **OSCD-pretrained and Melbourne-fine-tuned Dual-Stream model achieving the best result among the six tested configurations**.

This suggests that pretrained representations learned from a broader change-detection dataset can provide a useful starting point when developing a change-detection model for a new geographic environment with limited local labelled data.

The project also demonstrates how model predictions can support an iterative process of **local ground-truth development and refinement**.

---

# 📌 Project Status

🚧 **Active research and portfolio project**

Current focus:

* Melbourne ground-truth refinement
* model comparison
* transfer-learning analysis
* visual validation
* reproducibility and documentation

---

## 👤 Author

**Tia Azarm**

Geospatial Analyst | GIS & Spatial Analysis | Remote Sensing | GeoAI

Melbourne, Victoria, Australia
