import os
import pydicom
from PIL import Image
import numpy as np

def convert_dcm_to_jpg(dcm_path, output_folder):
    # Load DICOM file
    ds = pydicom.dcmread(dcm_path)

    # Check if the DICOM file contains an image
    if 'PixelData' in ds:
        # Convert pixel data to numpy array
        img = ds.pixel_array

        # Normalize pixel values to 0-255
        img = img.astype(np.float32)
        img = (img / np.max(img)) * 255
        img = img.astype(np.uint8)

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Generate output file path
        output_path = os.path.join(output_folder, os.path.basename(dcm_path)[:-4] + '.jpg')

        # Convert and save as JPEG
        img_pil = Image.fromarray(img)
        img_pil.save(output_path)
        print(f"File saved as: {output_path}")
    else:
        print(f"No image found in {dcm_path}")

# Example usage:
input_folder = '/content/drive/MyDrive/datasets/DCM_Images/'
output_folder = '/content/drive/MyDrive/datasets/JPG_Images/'

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".dcm"):
        dcm_path = os.path.join(input_folder, filename)
        convert_dcm_to_jpg(dcm_path, output_folder)