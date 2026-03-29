# CT Preprocessing Pipeline for Lung Imaging

## Overview

This repository contains preprocessing code developed as part of a thesis project on lung CT analysis.
It provides a structured pipeline to prepare raw CT scans for downstream deep learning tasks such as lung segmentation and nodule detection.

---

## Pipeline Components

The preprocessing workflow includes:

* **Resampling**
  Standardizing voxel spacing across CT scans to ensure consistency.

* **Lung ROI Extraction**
  Isolating the lung region using segmentation masks to remove irrelevant background.

* **Intensity Normalization**
  Adjusting CT intensity values for stable model training.

* **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
  Enhancing local contrast to improve visibility of small structures such as nodules.

---

## Repository Structure

```text
ct-preprocessing-pipeline/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── resampling/
│   ├── roi_extraction/
│   ├── normalization/
│   ├── clahe/
│   └── utils/
│
├── notebooks/
│   └── preprocessing_demo.ipynb
│
└── results/
    └── figures/
```

---

## Usage (Basic)

1. Prepare input CT scans (e.g., `.mhd`, `.nii`, or `.npy`)
2. Apply preprocessing steps in sequence:

   * Resampling
   * Lung ROI extraction
   * Normalization
   * CLAHE
3. Save processed volumes for training

Detailed scripts and examples are available in the `src/` and `notebooks/` directories.

---

## Notes

* This repository is part of a broader pipeline for lung CT analysis and segmentation.
* Dataset files are not included. Please refer to public datasets such as **LUNA16** or **LIDC-IDRI**.

---

## Status

Initial version. Code structure and documentation will be improved over time.
