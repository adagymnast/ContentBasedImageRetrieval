# Import libraries
import numpy as np
import cv2
from imutils import paths
import time
import os
import math
import matplotlib.pyplot as plt
import skimage

# Fixed constants
ALGORITHM = 'shape_daisy'

# Parameters
DIMENSION = (50, 50)
PARAM = ''

# Datasets
DATASETS = ['groundtruth', 'wang', 'art']
DATASET_NAME = 'wang'
EXTENSION = '.npy'

# Relative paths
ABSOLUTE_PATH = os.path.abspath(__file__)
FILE_DIRECTORY = os.path.dirname(ABSOLUTE_PATH)
ROOT_DIRECTORY = os.path.dirname(FILE_DIRECTORY)

# Dataset
INPUT_DATASET_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Datasets')
INPUT_DATASET_PATH = os.path.join(INPUT_DATASET_DIRECTORY, DATASET_NAME)

# Features
OUTPUT_FEATURES_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Features')

# Time
START_TIME = time.time()


if __name__ == '__main__':

    Images_Paths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    Images_Count = len(Images_Paths)

    # Test image: for Feature_Count
    Image_Test = cv2.resize(cv2.imread(Images_Paths[0], cv2.IMREAD_GRAYSCALE),  DIMENSION)
    Daisy_Test = skimage.feature.daisy(
            Image_Test, step=4, radius=15, rings=3, histograms=8, orientations=8,
            normalization='l1', sigmas=None, ring_radii=None, visualize=False).reshape(-1)
    Features_Count = len(Daisy_Test)

    # Daisy Features
    Daisy_Features = np.zeros((Images_Count, Features_Count))
    print(Daisy_Features.shape)

    for image_index, image_path in enumerate(Images_Paths):
        Image_Gray = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE),  DIMENSION)
        Daisy_Features[image_index] = skimage.feature.daisy(
            Image_Gray, step=4, radius=15, rings=3, histograms=8, orientations=8,
            normalization='l1', sigmas=None, ring_radii=None, visualize=False).reshape(-1)

        print(image_index + 1, 'of', Images_Count, "--- %s seconds ---" % (time.time() - START_TIME))

    OUTPUT_FEATURES_FILE = 'features_' + ALGORITHM + PARAM + DATASET_NAME + EXTENSION
    OUTPUT_FEATURES_PATH = os.path.join(OUTPUT_FEATURES_DIRECTORY, OUTPUT_FEATURES_FILE)

    # Save features as CSV file
    np.save(OUTPUT_FEATURES_PATH, Daisy_Features)
    print("All " + ALGORITHM + " computation took", "--- %s seconds ---" % (time.time() - START_TIME))
