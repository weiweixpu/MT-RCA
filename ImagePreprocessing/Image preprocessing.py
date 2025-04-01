
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import os

# Resample the image to a new spacing
def resample_image(itk_image, out_spacing=[1.0, 1.0, 1.0]):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled_image = resample.Execute(itk_image)
    return resampled_image

# Find the first non-zero point in a mask image
def find_unique_nonzero_point(mask_image):
    mask_array = sitk.GetArrayFromImage(mask_image)
    nonzero_indices = np.transpose(np.nonzero(mask_array))
    if len(nonzero_indices) > 0:
        return tuple(nonzero_indices[0][::-1])
    else:
        return None

# Crop a cube around a given center point
def crop_cube_around_point(input_path, output_path, center_xyz, cube_size):
    nifti_img = nib.load(input_path)
    img_data = nifti_img.get_fdata()

    start_xyz = np.maximum(center_xyz - cube_size // 2, 0)
    end_xyz = np.minimum(center_xyz + cube_size // 2, img_data.shape)
    pad_start = np.maximum(0, cube_size // 2 - center_xyz)
    pad_end = np.maximum(0, center_xyz + cube_size // 2 - np.array(img_data.shape))

    cropped_data = np.pad(img_data[start_xyz[0]:end_xyz[0], start_xyz[1]:end_xyz[1], start_xyz[2]:end_xyz[2]],
                          ((pad_start[0], pad_end[0]), (pad_start[1], pad_end[1]), (pad_start[2], pad_end[2])),
                          mode='constant', constant_values=0)

    cropped_nifti_img = nib.Nifti1Image(cropped_data, affine=nifti_img.affine)
    nib.save(cropped_nifti_img, output_path)

# Define the folder path
folder_path = r'/data/xiaoyingzhen/test12'

# Process each subfolder
for subfolder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        print('Processing folder:', subfolder)

        gz_path = os.path.join(subfolder_path, 'original.nii.gz')
        mask_path = os.path.join(subfolder_path, 'dian_Untitled.nii.gz')

        Original_img = sitk.ReadImage(gz_path)
        Mask_img = sitk.ReadImage(mask_path)

        # Find unique non-zero point in the original mask
        unique_point_original = find_unique_nonzero_point(Mask_img)
        if unique_point_original is not None:
            print('Unique nonzero point in original mask:', unique_point_original)

        # Resample images
        Resample_img = resample_image(Original_img)
        Resample_mask = resample_image(Mask_img)

        # Save resampled images
        output_image_path = os.path.join(subfolder_path, 'resampled_image.nii.gz')
        output_mask_path = os.path.join(subfolder_path, 'resampled_mask.nii.gz')

        sitk.WriteImage(Resample_img, output_image_path)
        sitk.WriteImage(Resample_mask, output_mask_path)

        # Find unique non-zero point in the resampled mask
        unique_point = find_unique_nonzero_point(Resample_mask)
        if unique_point is not None:
            print('Unique nonzero point in resampled mask:', unique_point)

        # Estimate the new coordinates of the unique point after resampling
        new_spacing = Resample_mask.GetSpacing()
        estimated_point = tuple(int(np.round(coord / new_spacing[i] * Original_img.GetSpacing()[i])) for i, coord in
                                enumerate(unique_point_original))

        print('Estimated coordinates after resampling:', estimated_point)

        # Define cropping center and size
        center_coordinates = estimated_point
        cube_size = np.array([224, 224, 224])

        input_nifti_path = output_image_path
        subfolder_name = os.path.basename(subfolder_path)
        output_nifti_filename = f"{subfolder_name}.nii.gz"
        output_nifti_path = os.path.join(folder_path, output_nifti_filename)

        # Crop and save the result
        crop_cube_around_point(input_nifti_path, output_nifti_path, center_coordinates, cube_size)
        print('Cropped output saved to:', output_nifti_path)
