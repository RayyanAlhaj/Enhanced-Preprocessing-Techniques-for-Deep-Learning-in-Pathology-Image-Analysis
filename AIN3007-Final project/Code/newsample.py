import os
import openslide
import numpy as np
import xml.etree.ElementTree as ET
import cv2

# Configuration
WSI_DIR = '/Volumes/moedisk/AIN3007_project/train_valid_data/Breast1__he'
OUTPUT_IMAGES_DIR = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/Downsampledslides/images'
OUTPUT_MASKS_DIR = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/Downsampledslides/masks'

DOWNSAMPLE_SIZE = 1024

def parse_xml_annotations(xml_path):
    """
    Parse XML file to extract ROI coordinates.
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

def get_roi_bounding_box(rois):
    """
    Calculate the bounding box of the ROI.
    """
    x_coordinates = [point[0] for roi in rois for point in roi]
    y_coordinates = [point[1] for roi in rois for point in roi]
    min_x, max_x = min(x_coordinates), max(x_coordinates)
    min_y, max_y = min(y_coordinates), max(y_coordinates)
    return min_x, min_y, max_x, max_y

def create_downsampled_mask(slide, rois, size, bounding_box):
    """
    Create a downsampled mask from a WSI based on ROI annotations.
    """
    scale_factor = slide.level_downsamples[0]
    scaled_rois = [[(int(x / scale_factor), int(y / scale_factor)) for x, y in roi] for roi in rois]
    mask = np.zeros((size, size), dtype=np.uint8)

    for roi in scaled_rois:
        cv2.fillPoly(mask, [np.array(roi, dtype=np.int32)], color=255)

    return mask

def process_wsi_and_annotation(wsi_path, xml_path, output_image_dir, output_mask_dir, size):
    """
    Process a single WSI and its corresponding XML annotation.
    """
    try:
        slide = openslide.OpenSlide(wsi_path)
        rois = parse_xml_annotations(xml_path)
        bounding_box = get_roi_bounding_box(rois)

        scale_factor = slide.level_downsamples[0]
        scaled_bounding_box = tuple(int(coord / scale_factor) for coord in bounding_box)

        # Extract ROI region
        roi_region = slide.read_region(scaled_bounding_box[:2], 0, (scaled_bounding_box[2] - scaled_bounding_box[0], scaled_bounding_box[3] - scaled_bounding_box[1]))
        roi_image = np.array(roi_region)[:, :, 0:3]

        # Resize to desired output size
        resized_roi_image = cv2.resize(roi_image, (size, size), interpolation=cv2.INTER_LINEAR)

        # Create mask
        mask = create_downsampled_mask(slide, rois, size, bounding_box)

        # Save the downsampled image and mask
        base_name = os.path.splitext(os.path.basename(wsi_path))[0]
        cv2.imwrite(os.path.join(output_image_dir, f"{base_name}.png"), resized_roi_image)
        cv2.imwrite(os.path.join(output_mask_dir, f"{base_name}_mask.png"), mask)

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