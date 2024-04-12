import cv2
import os
import numpy as np

# Directories
ROI_PATCHES_DIR = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/samplepatch/roi'
INSPECTION_DIR = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/samplepatch/inspection'

# Configure thresholds
COLOR_VARIANCE_THRESHOLD = 2
WHITE_THRESHOLD = 240
PERCENTAGE_WHITE_THRESHOLD = 90

def check_color_variance(patch, threshold=COLOR_VARIANCE_THRESHOLD):
    return np.std(patch) > threshold

def check_whitespaces(patch, white_threshold=WHITE_THRESHOLD, percentage_threshold=PERCENTAGE_WHITE_THRESHOLD):
    white_pixels = np.sum(np.all(patch > white_threshold, axis=2))
    total_pixels = patch.shape[0] * patch.shape[1]
    return (white_pixels / total_pixels) * 100 < percentage_threshold

def is_patch_valid(patch):
    return check_color_variance(patch) and check_whitespaces(patch)

def process_patches(directory, inspection_dir):
    if not os.path.exists(inspection_dir):
        os.makedirs(inspection_dir)
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add more formats if needed
            patch_path = os.path.join(directory, filename)
            patch = cv2.imread(patch_path)

            if not is_patch_valid(patch):
                os.rename(patch_path, os.path.join(inspection_dir, filename))

def main():
    process_patches(ROI_PATCHES_DIR, INSPECTION_DIR)

if __name__ == '__main__':
    main()
    
