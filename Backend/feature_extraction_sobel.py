# Import libraries
import numpy as np
import cv2 as cv
from imutils import paths
import time
import os

# Fixed constants
ALGORITHM = 'shape_sobel'

# Parameters: 2 directions for Sobel operator
QUANTIZE = 64
KERNEL_COUNT = 2
HISTOGRAM_LEN = KERNEL_COUNT * QUANTIZE
PARAM = str(QUANTIZE)

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


def compute_sobel(img_gray, max_count=256):
    """
    Apply Sobel operator of the grayscale image
    :param img_gray: Grayscale image
    :return: Image normalized gradient in x and in y
    """
    gradient_x = cv.Sobel(img_gray, -1, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    gradient_y = cv.Sobel(img_gray, -1, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    gradient_x_normalized = cv.convertScaleAbs(gradient_x) / max_count
    gradient_y_normalized = cv.convertScaleAbs(gradient_y) / max_count
    return gradient_x_normalized, gradient_y_normalized


def compute_histogram(img_normalized, quantize):
    """
    Compute normalized histogram from normalized
    :param quantize: Number of representative edge intensities
    :param img_normalized: Normalized edge image computed by Sobel
    :return:Normalized histogram of edges
    """
    rows, cols = img_normalized.shape
    img_size = rows * cols

    # Uniform quantization
    img_quantized = (np.floor(quantize * img_normalized).astype(int) / quantize)
    img_quantized_indices = (img_quantized * quantize).astype(int)

    # Compute histogram
    uniques, counts = np.unique(img_quantized_indices, return_counts=True)
    histogram = np.zeros(shape=quantize)
    for index, count in zip(uniques, counts):
        histogram[index] = count

    return histogram / img_size


if __name__ == '__main__':
    Images_Paths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    Images_Count = len(Images_Paths)
    Features = np.zeros((Images_Count, HISTOGRAM_LEN))

    for image_index, image_path in enumerate(Images_Paths):
        Image = cv.imread(image_path)
        Gray = cv.cvtColor(Image, cv.COLOR_BGR2GRAY)

        # Compute image gradient using Sobel operator
        Gradient_X, Gradient_Y = compute_sobel(Gray)

        # Compute edge histogram
        Edge_Histogram_X = compute_histogram(Gradient_X, QUANTIZE)
        Edge_Histogram_Y = compute_histogram(Gradient_Y, QUANTIZE)

        # Save features
        Features[image_index] = np.concatenate((Edge_Histogram_X, Edge_Histogram_Y))
        print(image_index + 1, 'of', Images_Count, "--- %s seconds ---" % (time.time() - START_TIME))

    # Saving color histogram features: numpy array as csv file
    np.save(OUTPUT_FEATURES_PATH, Features)
    print("All " + ALGORITHM + " computation took", "--- %s seconds ---" % (time.time() - START_TIME))
