# data_prep.py
import os
import random
import openslide
import numpy as np
import xml.etree.ElementTree as ET
import cv2


'''
THIS FILE WAS MADE JUST TO EXTRACT RANDOM PATCHES FROM SLIDES
TO MAKE A POOL TO SELECT A REFERENCE IMAGE FOR STAIN 
NORMALIZATION. 
THIS WILL HAVE NO FURTHER USE AFTER SELECTING THE REFERENCE
IMAGE. (it is selected)
'''

WSI_IMAGES_DIR = '/Volumes/moedisk/AIN3007_project/train_valid_data/LymphNode1__he'
XML_ANNOTATIONS_DIR = '/Volumes/moedisk/AIN3007_project/train_valid_data/LymphNode1__he'
ROI_PATCHES_DIR = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/random_patches'
PATCH_SIZE = 512  # or any other size you want


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


def extract_random_patches_from_wsi(slide, slide_filename, rois, output_dir, patch_size, level=0, num_patches=4):
    """
    Extract random patches from a WSI at a specified level based on ROI coordinates.
    """
    downsample = slide.level_downsamples[level]
    for i, roi in enumerate(rois):
        scaled_roi = [(x / downsample, y / downsample) for x, y in roi]
        x, y, w, h = cv2.boundingRect(np.array(scaled_roi, dtype=np.int32))

        patches_extracted = 0
        while patches_extracted < num_patches:
            rand_x = random.randint(x, x + w - patch_size)
            rand_y = random.randint(y, y + h - patch_size)
            patch = slide.read_region((int(rand_x * downsample), int(rand_y * downsample)), level, (patch_size, patch_size))
            patch = np.array(patch)[:, :, 0:3]  # Convert to RGB
            patch_name = f'{os.path.splitext(slide_filename)[0]}_roi_{i}_patch_{patches_extracted}.png'
            cv2.imwrite(os.path.join(output_dir, patch_name), patch)
            patches_extracted += 1

def process_wsi_and_annotation(wsi_path, xml_path, output_dir):
    """
    Process a single WSI and its corresponding XML annotation.
    """
    slide = openslide.OpenSlide(wsi_path)
    rois = parse_xml_annotations(xml_path)
    slide_filename = os.path.basename(wsi_path)
    extract_random_patches_from_wsi(slide, slide_filename, rois, output_dir, PATCH_SIZE)

def main():
    # Check if output directory exists, if not, create it
    if not os.path.exists(ROI_PATCHES_DIR):
        os.makedirs(ROI_PATCHES_DIR)

    for filename in os.listdir(WSI_IMAGES_DIR):
        if filename.lower().endswith(('.svs', '.tif', '.tiff', '.mrxs', '.ndpi')):
            wsi_path = os.path.join(WSI_IMAGES_DIR, filename)
            xml_filename = os.path.splitext(filename)[0] + '.xml'
            xml_path = os.path.join(XML_ANNOTATIONS_DIR, xml_filename)

            if os.path.exists(xml_path):
                try:
                    process_wsi_and_annotation(wsi_path, xml_path, ROI_PATCHES_DIR)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
            else:
                print(f"XML annotation not found for {filename}")

if __name__ == '__main__':
    main()