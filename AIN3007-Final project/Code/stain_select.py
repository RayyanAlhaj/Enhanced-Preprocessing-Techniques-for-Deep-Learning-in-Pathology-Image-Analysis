import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

'''
THIS CODE WILL SELECT THE MOST SUITABLE IMAGE 
TO BE A REFERENCE IMAGE FOR STAIN NORMALIZATION.
THIS WILL HAVE NO FURTHER USE AFTER SELECTING 
THE REFERENCE IMAGE. (it is selected)
'''

PATCHES_DIR = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/random_patches'  # Directory containing the patches
REFERENCE_PATCH_DIR = '/Users/hamodyx0/Documents/school/deeplearning/DLproject/'  # Directory to save the reference patch
NUM_CLUSTERS = 5  # Number of clusters to form
NUM_COMPONENTS = 3  # Number of PCA components for color feature reduction

def extract_color_features(image):
    """ Extract color features (mean and standard deviation) from the image """
    mean, std = cv2.meanStdDev(image)
    return np.concatenate([mean.flatten(), std.flatten()])

def perform_clustering(features):
    """ Perform K-means clustering on the extracted features """
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    kmeans.fit(features)
    return kmeans

def select_reference_image(filenames, kmeans):
    """ Select a reference image from the most populous cluster """
    cluster_counts = np.bincount(kmeans.labels_)
    most_populous_cluster = np.argmax(cluster_counts)

    for i, label in enumerate(kmeans.labels_):
        if label == most_populous_cluster:
            return filenames[i]  # Return filename of the first patch from this cluster

def main():
    features = []
    filenames = []

    # Extract color features from each patch
    for filename in os.listdir(PATCHES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(PATCHES_DIR, filename))
            features.append(extract_color_features(img))
            filenames.append(filename)

    # Reduce feature dimensionality
    pca = PCA(n_components=NUM_COMPONENTS)
    reduced_features = pca.fit_transform(np.array(features))

    # Perform clustering
    kmeans = perform_clustering(reduced_features)

    # Select reference image
    reference_image = select_reference_image(filenames, kmeans)
    print(f"Selected reference image: {reference_image}")

if __name__ == '__main__':
    main()