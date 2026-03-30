
# ROI Extraction and CT Preparation

This module prepares lung CT volumes for downstream 2D and 3D deep learning workflows.

## What this code does

The pipeline performs the following steps:

1. Windowing
   CT intensities are clipped to a lung-specific range and normalized.  
   This reduces irrelevant intensity variation and makes the input more consistent.

2. Lung-masked CLAHE
   CLAHE is applied only inside the predicted lung region.  
   This improves local contrast while avoiding enhancement of irrelevant background areas.

3. Bounding box extraction
   A 3D bounding box is computed from the lung prediction mask to locate the lung region.

4. ROI cropping
   The CT volume and, when available, the lesion mask are cropped using the lung bounding box.  
   This reduces empty background and focuses the data on the relevant anatomy.

5. Saving original and resized outputs
   The cropped volumes are saved in:
   - original cropped size
   - resized format for model input

## Output format

The pipeline can save:

- cropped CT volume at original ROI size
- cropped lesion mask at original ROI size
- resized CT volume (for example 256 × 256 × depth)
- resized lesion mask using nearest-neighbor interpolation

## Why this step is important

This preprocessing stage reduces background content, enhances lung structures, and creates a more standardized input for both 2D and 3D segmentation models.

## Notes

- CT volumes are resized with linear interpolation.
- Masks are resized with nearest-neighbor interpolation to preserve label values.
- Lung predictions are used only to define the anatomical ROI, while the ground-truth lesion mask is cropped accordingly.
