
from pathlib import Path

import nibabel as nib
import numpy as np
import cv2 as cv
from skimage.exposure import equalize_adapthist
from skimage.transform import resize


def apply_window(volume, wmin=-1000, wmax=100):
    dump = data.copy()
    dump[dump > wmax] = wmax
    dump[dump < wmin] = wmin

    normalized = (dump - wmin) / (wmax - wmin)

    return normalized


def apply_clahe_3d(ct, Pred, clip_limit=0.007, kernel_size=(8, 8, 8)):

    Pred = Pred.astype(np.uint8)
    ct_roi = np.where(Pred, ct, 0)

    ct_roi = windower(ct_roi, -1000, 100)

    original_min = ct_roi.min()
    original_max = ct_roi.max()

    ct_roi = cv.normalize(ct_roi, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


    ct_clahe = equalize_adapthist(ct_roi, kernel_size=(8, 8, 8), clip_limit=0.007)


    ct_roi2 = cv.normalize(ct_clahe, None, original_min, original_max, cv.NORM_MINMAX)

    ct_enhanced = np.where(Pred, ct_roi2, 0)

    return ct_enhanced

def get_3d_bounding_box(mask):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Mask is empty. Cannot compute bounding box.")

    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)
    return x_min, x_max, y_min, y_max, z_min, z_max


def crop_to_lung_region(ct, lung_pred, mask, margin_xy=10):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox_3D(lung_pred)  # Use mask to define ROI

    # Apply cropping
    cropped_ct = ct[xmin:xmax+10, ymin:ymax+10, zmin:zmax]
    pr_ct = np.rot90(cropped_ct)
    pr_ct = np.flipud(pr_ct)

    cropped_mask = mask[xmin:xmax+10, ymin:ymax+10, zmin:zmax]
    pr_m = np.rot90(cropped_mask)
    pr_m = np.flipud(pr_m)

    return pr_ct, pr_m


def resize_ct_volume(ct, output_shape):
    return resize(
        ct,
        output_shape=output_shape,
        order=1,
        mode="constant",
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def resize_mask_volume(mask, output_shape):
    return resize(
        mask,
        output_shape=output_shape,
        order=0,
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    ).astype(mask.dtype)


def prepare_case(ct, lung_pred, Nodule_mask=None, resize_hw=256):
    enhanced_ct = apply_clahe_3d(ct, lung_mask)

    if lesion_mask is None:
        cropped_ct = crop_to_lung_region(enhanced_ct, lung_mask)
        depth = cropped_ct.shape[2]
        resized_ct = resize_ct_volume(cropped_ct, (resize_hw, resize_hw, depth))
        return cropped_ct, resized_ct

    cropped_ct, cropped_mask = crop_to_lung_region(
        enhanced_ct,
        lung_pred,
        Nodule_mask,
    )

    depth = cropped_ct.shape[2]
    resized_ct = resize_ct_volume(cropped_ct, (resize_hw, resize_hw, depth))
    resized_mask = resize_mask_volume(cropped_mask, (depth, resize_hw, resize_hw))

    return cropped_ct, cropped_mask, resized_ct, resized_mask


def load_nifti(path):
    return nib.load(str(path)).get_fdata()


def save_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def process_case(
    ct_path,
    lung_mask_path,
    output_dir,
    lesion_mask_path=None,
    prefix="",
    resize_hw=256,
):
    ct = load_nifti(ct_path)
    lung_pred = load_nifti(lung_mask_path)

    case_id = Path(ct_path).stem.replace(".nii", "")
    output_dir = Path(output_dir)

    if lesion_mask_path is None:
        cropped_ct, resized_ct = prepare_case(
            ct=ct,
            lung_pred=lung_pred,
            lesion_mask=None,
            resize_hw=resize_hw,
        )

        save_array(output_dir / "orgsize_CT" / f"{prefix}{case_id}.npy", cropped_ct)
        save_array(output_dir / "resized_CT_256" / f"{prefix}{case_id}.npy", resized_ct)
        return

    lesion_mask = load_nifti(lesion_mask_path)

    cropped_ct, cropped_mask, resized_ct, resized_mask = prepare_case(
        ct=ct,
        lung_pred=lung_pred,
        lesion_mask=lesion_mask,
        resize_hw=resize_hw,
    )

    save_array(output_dir / "orgsize_CT" / f"{prefix}{case_id}.npy", cropped_ct)
    save_array(output_dir / "orgsize_MASK" / f"{prefix}{case_id}.npy", cropped_mask)
    save_array(output_dir / "resized_CT_256" / f"{prefix}{case_id}.npy", resized_ct)
    save_array(output_dir / "resized_MASK_256" / f"{prefix}{case_id}.npy", resized_mask)
