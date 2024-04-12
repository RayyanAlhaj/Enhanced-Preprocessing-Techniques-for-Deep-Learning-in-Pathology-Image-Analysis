import os
import cv2
import staintools
import numpy as np
import multiprocessing
from multiprocessing import Pool

# Load the reference image for stain normalization
REFERENCE_IMAGE_PATH = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/TCGA-B6-A0RQ-01Z-00-DX1_roi_0_patch_2.png'
reference_image = staintools.read_image(REFERENCE_IMAGE_PATH)
reference_image = staintools.LuminosityStandardizer.standardize(reference_image)

normalizer = staintools.StainNormalizer(method='macenko')
normalizer.fit(reference_image)

def adjust_intensity(image, factor):
    """ Adjust the intensity of the image based on the factor. """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = np.array(hsv_image, dtype=np.float64)
    hsv_image[:,:,2] = hsv_image[:,:,2]*factor
    hsv_image[:,:,2][hsv_image[:,:,2]>255] = 255
    hsv_image = np.array(hsv_image, dtype=np.uint8)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image

def apply_staining_threshold(image, threshold):
    """ Apply a threshold on the staining intensity. """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    masked_image = cv2.bitwise_and(image, image, mask=threshold_image)
    return masked_image

def process_image(image_path, output_dir, intensity_factor, staining_threshold):
    try:
        original_image = staintools.read_image(image_path)
        original_image = staintools.LuminosityStandardizer.standardize(original_image)

        # Normalize the image
        normalized_image = normalizer.transform(original_image)

        # Adjust intensity
        normalized_image = adjust_intensity(normalized_image, intensity_factor)

        # Apply staining threshold
        normalized_image = apply_staining_threshold(normalized_image, staining_threshold)

        # Save the image
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_image_path, normalized_image)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        
def normalize_stains(input_path, output_path, intensity_factor, staining_threshold):
    # Create a list of all image paths
    image_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_path) for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Prepare arguments for multiprocessing
    args = [(image_path, output_path, intensity_factor, staining_threshold) for image_path in image_paths]

    # Set up multiprocessing
    pool = Pool(multiprocessing.cpu_count())
    pool.starmap(process_image, args)
    pool.close()
    pool.join()

if __name__ == '__main__':
    input_base_dir = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/Downsampledslides/images'
    output_base_dir = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/stain_output'
    intensity_factor = 1.3  # Adjust as needed
    staining_threshold = 200  # Adjust as needed (0-255 for grayscale)
    normalize_stains(input_base_dir, output_base_dir, intensity_factor, staining_threshold)
