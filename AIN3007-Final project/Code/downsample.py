import os
import openslide
import numpy as np
import xml.etree.ElementTree as ET
import cv2

# Configuration
WSI_DIR = '/Volumes/moedisk/AIN3007_project/train_valid_data/Breast3__ihc'
OUTPUT_IMAGES_DIR = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/Downsampledslides/images'
OUTPUT_MASKS_DIR = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/Downsampledslides/masks'

TARGET_SIZE = 1024


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


def resize_with_padding(image, target_size):
    """
    Resize an image to the target size with padding to maintain aspect ratio.
    """
    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # New size should be in (width, height) format
    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = target_size - new_size[1]
    delta_h = target_size - new_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image

def downsample_image_and_mask(slide, rois, target_size):
    """
    Downsample the WSI and create a mask of the same size based on ROI annotations.
    """
    # Downsample the slide
    downsampled_image = np.array(slide.get_thumbnail((target_size, target_size)))[:, :, 0:3]

    # Calculate scale factors
    scale_x = slide.level_dimensions[0][0] / downsampled_image.shape[1]
    scale_y = slide.level_dimensions[0][1] / downsampled_image.shape[0]

    # Create mask
    mask = np.zeros(downsampled_image.shape[:2], dtype=np.uint8)

    for roi in rois:
        scaled_roi = [(int(x / scale_x), int(y / scale_y)) for x, y in roi]
        cv2.fillPoly(mask, [np.array(scaled_roi, dtype=np.int32)], color=255)

    # Resize both image and mask
    resized_image = resize_with_padding(downsampled_image, target_size)
    resized_mask = resize_with_padding(mask, target_size)

    return resized_image, resized_mask

def process_wsi_and_annotation(wsi_path, xml_path, output_image_dir, output_mask_dir, target_size):
    """
    Process a single WSI and its corresponding XML annotation.
    """
    try:
        slide = openslide.OpenSlide(wsi_path)
        rois = parse_xml_annotations(xml_path)
        base_name = os.path.splitext(os.path.basename(wsi_path))[0]
        output_image_path = os.path.join(output_image_dir, base_name + '.png')
        output_mask_path = os.path.join(output_mask_dir, base_name + '_mask.png')

        # Downsample image and mask
        resized_image, resized_mask = downsample_image_and_mask(slide, rois, target_size)

        # Save the resized image and mask
        cv2.imwrite(output_image_path, resized_image)
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
                process_wsi_and_annotation(wsi_path, xml_path, OUTPUT_IMAGES_DIR, OUTPUT_MASKS_DIR, TARGET_SIZE)

if __name__ == '__main__':
    main()
