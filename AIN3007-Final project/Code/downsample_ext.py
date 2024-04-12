import os
import openslide
import numpy as np
import xml.etree.ElementTree as ET
import cv2


'''
This code extracts a 1024x1024 pixel images from the WSI slides,
and it extracts the mask according to the annotations. 

How it works is, it runs through a directory that contains both the 
slides and their annotations (which share the same titles), and 
extract the slides and the masks to seperate files.

you can adjust the downsample size.

Since this dataset is small, we can use heavy data augementations.
'''

# Configuration
WSI_DIR = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/sample_data'
OUTPUT_IMAGES_DIR = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/down_sampled/images'
OUTPUT_MASKS_DIR = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/down_sampled/masks'
DOWNSAMPLE_SIZE = 1024

def parse_xml_annotations(xml_path):
    """
    Parse XML file to extract ROI coordinates, handling different decimal separators.
    """
    rois = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for annotation in root.findall('.//Annotation'):
        points = []
        for point in annotation.findall('.//Coordinate'):
            x = float(point.get('X').replace(',', '.'))
            y = float(point.get('Y').replace(',', '.'))
            points.append((x, y))
        rois.append(points)

    return rois

def create_downsampled_mask(slide, rois, size):
    """
    Create a downsampled mask from a WSI based on ROI annotations.
    """
    scale_factor = max(slide.level_dimensions[0]) / size
    downsampled_dims = (int(slide.level_dimensions[0][1] / scale_factor), int(slide.level_dimensions[0][0] / scale_factor))
    mask = np.zeros(downsampled_dims, dtype=np.uint8)

    for roi in rois:
        scaled_roi = [(int(x / scale_factor), int(y / scale_factor)) for x, y in roi]
        cv2.fillPoly(mask, [np.array(scaled_roi, dtype=np.int32)], color=255)

    return mask

def process_wsi_and_annotation(wsi_path, xml_path, output_image_dir, output_mask_dir, size):
    """
    Process a single WSI and its corresponding XML annotation.
    """
    try:
        slide = openslide.OpenSlide(wsi_path)
        rois = parse_xml_annotations(xml_path)
        base_name = os.path.splitext(os.path.basename(wsi_path))[0]
        output_image_path = os.path.join(output_image_dir, base_name + '.png')
        output_mask_path = os.path.join(output_mask_dir, base_name + '_mask.png')

        # Create downsampled mask
        mask = create_downsampled_mask(slide, rois, size)

        # Downsample the image
        downsampled_image = np.array(slide.get_thumbnail((size, size)))[:, :, 0:3]

        # Resize the mask to match the image size
        resized_mask = cv2.resize(mask, (downsampled_image.shape[1], downsampled_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Save the downsampled image and resized mask
        cv2.imwrite(output_image_path, downsampled_image)
        cv2.imwrite(output_mask_path, resized_mask)
    except Exception as e:
        print(f"Error processing {wsi_path}: {e}")

def main():
    if not os.path.exists(OUTPUT_IMAGES_DIR):
        os.makedirs(OUTPUT_IMAGES_DIR)
    if not os.path.exists(OUTPUT_MASKS_DIR):
        os.makedirs(OUTPUT_MASKS_DIR)

    for filename in os.listdir(WSI_DIR):
        if filename.lower().endswith(('.svs', '.tif', '.tiff', '.mrxs', '.ndpi')):
            wsi_path = os.path.join(WSI_DIR, filename)
            xml_filename = os.path.splitext(filename)[0] + '.xml'
            xml_path = os.path.join(WSI_DIR, xml_filename)

            if os.path.exists(xml_path):
                process_wsi_and_annotation(wsi_path, xml_path, OUTPUT_IMAGES_DIR, OUTPUT_MASKS_DIR, DOWNSAMPLE_SIZE)

if __name__ == '__main__':
    main()
