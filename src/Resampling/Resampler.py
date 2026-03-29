
import SimpleITK as sitk
import numpy as np


def resample_to_isotropic(image, spacing=1.0):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / spacing)))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([spacing] * 3)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


def resample_mask(mask, ct_image, reference_image, spacing=(1.0, 1.0, 1.0)):
    sitk_mask = sitk.GetImageFromArray(mask)
    sitk_mask.SetSpacing(ct_image.GetSpacing())
    sitk_mask.SetOrigin(ct_image.GetOrigin())
    sitk_mask.SetDirection(ct_image.GetDirection())

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(reference_image.GetSize())
    resampler.SetOutputDirection(reference_image.GetDirection())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled_mask = resampler.Execute(sitk_mask)
    resampled_mask = sitk.GetArrayFromImage(resampled_mask)

    return (resampled_mask > 0).astype(np.uint8)


def resample_ct_and_save(input_path, output_path):
    image = sitk.ReadImage(input_path)
    resampled = resample_to_isotropic(image)
    array = sitk.GetArrayFromImage(resampled)
    np.save(output_path, array)
