# data_prep.py
import os
import openslide
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from config import WSI_IMAGES_DIR, XML_ANNOTATIONS_DIR, ROI_PATCHES_DIR, BACKGROUND_PATCHES_DIR, PATCH_SIZE

'''
In this code we read then extract patches
from the WSI slides. We extract the patches
from the Region of interest according to 
the .xml annotations. The reason why we dont
extract background patches is that we already
have sufficient background patches to learn 
from.
'''

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

def extract_patches_from_wsi(slide, rois, output_dir, patch_size, level=0):
    """
    Extract patches from a WSI at a specified level based on ROI coordinates.
    """
    downsample = slide.level_downsamples[level]
    for i, roi in enumerate(rois):
        scaled_roi = [(x / downsample, y / downsample) for x, y in roi]
        x, y, w, h = cv2.boundingRect(np.array(scaled_roi, dtype=np.int32))

        for row in range(y, y + h, patch_size):
            for col in range(x, x + w, patch_size):
                patch = slide.read_region((int(col * downsample), int(row * downsample)), level, (patch_size, patch_size))
                patch = np.array(patch)[:, :, 0:3]  # Drop the alpha channel
                cv2.imwrite(os.path.join(output_dir, f'roi_{i}_{row}_{col}.png'), patch)

def process_wsi_and_annotation(wsi_path, xml_path):
    """
    Process a single WSI and its corresponding XML annotation.
    """
    slide = openslide.OpenSlide(wsi_path)
    rois = parse_xml_annotations(xml_path)

    extract_patches_from_wsi(slide, rois, ROI_PATCHES_DIR, PATCH_SIZE)

    # Extract Background patches (similar to the previous implementation)
    # Make sure to adapt this part to use openslide as well

# ... [previous code remains unchanged]

def main():
    for filename in os.listdir(WSI_IMAGES_DIR):
        if filename.lower().endswith(('.svs', '.tif', '.tiff', '.mrxs', '.ndpi')):  # Include .ndpi format
            wsi_path = os.path.join(WSI_IMAGES_DIR, filename)
            xml_filename = os.path.splitext(filename)[0] + '.xml'
            xml_path = os.path.join(XML_ANNOTATIONS_DIR, xml_filename)

            if os.path.exists(xml_path):
                try:
                    process_wsi_and_annotation(wsi_path, xml_path)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
            else:
                print(f"XML annotation not found for {filename}")

if __name__ == '__main__':
    main()
