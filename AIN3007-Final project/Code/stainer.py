
import os
import cv2
import staintools
import multiprocessing
from multiprocessing import Pool

'''
This code's multiprocess implementation is made specifically for 
M1 macos systems, there will be another file attached 
for a multiprocessed staining code for non-m1 systems.
'''

# Load the reference image for stain normalization
REFERENCE_IMAGE_PATH = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/TCGA-B6-A0RQ-01Z-00-DX1_roi_0_patch_2.png'
reference_image = staintools.read_image(REFERENCE_IMAGE_PATH)
reference_image = staintools.LuminosityStandardizer.standardize(reference_image)

normalizer = staintools.StainNormalizer(method='macenko')
normalizer.fit(reference_image)

def process_image(image_path, output_dir):
    """Function to process each image."""
    original_image = staintools.read_image(image_path)
    original_image = staintools.LuminosityStandardizer.standardize(original_image)

    # Normalize and save the image
    normalized_image = normalizer.transform(original_image)
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_image_path, normalized_image)

def normalize_stains(input_path, output_path):
    # Create a list of all image paths
    image_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_path) for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Prepare arguments for multiprocessing
    args = [(image_path, output_path) for image_path in image_paths]

    # Set up multiprocessing
    pool = Pool(multiprocessing.cpu_count())
    pool.starmap(process_image, args)
    pool.close()
    pool.join()

if __name__ == '__main__':
    input_base_dir = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/Downsampledslides/images'
    output_base_dir = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/stain_output'
    normalize_stains(input_base_dir, output_base_dir)

