import os
import cv2
import numpy as np
from config import PATCHES_DIR


'''
note: shall be updated, them comments. 
What tis code does is that in runs through the directory
of patches -can be any directory- and check if an image is
qual or more than 85% white, if it is, it moves the image
to a different folder. It can also be modified to delete 
these images. I have made it move the images just for the
purpose of testing and inspecting them.
'''
# actually we can implement a function to the end model that would automatically 
# label a certian patch as background just by the background threshold
# actually this might involve some patches from the middle of a tissue, which
# might not be a good idea. To be continued...



# Directories
ROI_PATCHES_DIR = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/samplepatch/roi'
INSPECTION_DIR = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/samplepatch/inspection'
# Configure thresholds (these may need fine-tuning)
#WHITE_THRESHOLD = 230          # Define based on dataset inspection
#PERCENTAGE_WHITE_THRESHOLD = 20 # Define based on dataset inspection
# Configure thresholds
COLOR_VARIANCE_THRESHOLD = 3.3  # Define based on dataset inspection
INTENSITY_THRESHOLD = 10       # Determine through inspection
AREA_THRESHOLD = 70            # Minimal area to be considered tissue


def calculate_color_variance(patch):
    """ Calculate the variance of color in the patch. """
    mean, std_dev = cv2.meanStdDev(patch)
    return max(std_dev)

def check_intensity(patch, threshold=INTENSITY_THRESHOLD):
    """ Check if the patch has enough intensity to be considered as tissue. """
    grayscale = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)
    return np.sum(binary) > AREA_THRESHOLD

def is_patch_candidate(patch):
    """ Determine if a patch is likely to be tissue based on variance and intensity. """
    return calculate_color_variance(patch) > COLOR_VARIANCE_THRESHOLD and check_intensity(patch)

def process_patches(directory, inspection_dir):
    if not os.path.exists(inspection_dir):
        os.makedirs(inspection_dir)
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            patch_path = os.path.join(directory, filename)
            patch = cv2.imread(patch_path)

            if not is_patch_candidate(patch):
                os.rename(patch_path, os.path.join(inspection_dir, filename))

def main():
    process_patches(PATCHES_DIR, INSPECTION_DIR)

if __name__ == '__main__':
    main()