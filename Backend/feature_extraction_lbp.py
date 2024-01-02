import skimage
import numpy as np
import cv2
import os
from imutils import paths
import time

# Fixed constants
ALGORITHM = 'texture_lbp'

# Parameters
NUMBER_OF_POINTS = 24
RADIUS = 3
PARAM = 'P' + str(NUMBER_OF_POINTS) + '_' + 'R' + str(RADIUS)

# Datasets
DATASETS = ['groundtruth', 'wang', 'art']
DATASET_NAME = 'patterns'
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
OUTPUT_FEATURES_FILE = 'features_' + ALGORITHM + PARAM + DATASET_NAME + EXTENSION
OUTPUT_FEATURES_PATH = os.path.join(OUTPUT_FEATURES_DIRECTORY, OUTPUT_FEATURES_FILE)

# Time
START_TIME = time.time()


class LocalBinaryPatterns:
    def __init__(self, n_points, radius):
        # store the number of points and radius
        self.n_points = n_points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation of the image, and then use the LBP representation
        # Build the histogram of patterns
        lbp = skimage.feature.local_binary_pattern(image, self.n_points, self.radius, method="uniform")
        # cv2.imshow("LBP", lbp)
        # cv2.waitKey(0)
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.n_points + 3), range=(0, self.n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist


if __name__ == '__main__':
    Images_Paths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    Images_Count = len(Images_Paths)

    # Features initialization
    Features_Count = NUMBER_OF_POINTS + 2
    LBP_Features = np.zeros((Images_Count, Features_Count))

    for image_index, image_path in enumerate(Images_Paths):
        # Load image
        Image = cv2.imread(image_path)
        Image_Gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

        # Compute LBP features
        LBP_Features[image_index] = LocalBinaryPatterns(NUMBER_OF_POINTS, RADIUS).describe(Image_Gray)
        print(image_index + 1, 'of', Images_Count, ' ', "--- %s seconds ---" % (time.time() - START_TIME))

    # Save features as CSV file
    np.save(OUTPUT_FEATURES_PATH, LBP_Features)
    print("All " + ALGORITHM + " computation took", "--- %s seconds ---" % (time.time() - START_TIME))
