# Import libraries
import numpy as np
import cv2
from imutils import paths
import time
import os

# Fixed constants
ALGORITHM = 'shape_robinson'

# Parameters: 8 directions for Robinson compass mask
QUANTIZE = 128
PARAM = str(QUANTIZE)
ROBINSON_KERNELS = np.array(
    [
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        [[0, -1, -2], [1, 0, -1], [2, 1, 0]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
    ]
)

KERNELS_COUNT = ROBINSON_KERNELS.shape[0]
# HISTOGRAM_LEN = QUANTIZE * KERNELS_COUNT

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


def compute_robinson(img_gray, max_count=256):
    """
    Apply Robinson operator of the grayscale image
    :param img_gray: Grayscale image
    :param max_count: Maximum number of intensities
    :return: Image normalized gradient in x and in y
    """
    filtered_images = np.zeros(shape=(ROBINSON_KERNELS.shape[0], img_gray.shape[0], img_gray.shape[1]))
    for index_kernel, robinson_kernel in enumerate(ROBINSON_KERNELS):
        filtered_images[index_kernel] = cv2.filter2D(img_gray, -1, robinson_kernel) / max_count
    return filtered_images


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
    Features = np.zeros((Images_Count, KERNELS_COUNT, QUANTIZE))

    for image_index, image_path in enumerate(Images_Paths):

        Image = cv2.imread(image_path)
        Gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        Images_Filtered = compute_robinson(Gray)
        for i, image_filtered in enumerate(Images_Filtered):
            Features[image_index][i] = compute_histogram(image_filtered, QUANTIZE)
        print(image_index + 1, 'of', Images_Count, "--- %s seconds ---" % (time.time() - START_TIME))
    Features = Features.reshape(Images_Count, -1)

    # Saving color histogram features: numpy array as csv file
    np.save(OUTPUT_FEATURES_PATH, Features)
    print("All " + ALGORITHM + " computation took", "--- %s seconds ---" % (time.time() - START_TIME))
