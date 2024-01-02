# Import libraries
import numpy as np
import cv2
from imutils import paths
import time
import os

# Fixed constants
ALGORITHM = 'color_histogram'

# Parameters
COLOR_SPACES = ['RGB', 'HSV']
QUANTIZE = [8, 8, 8]

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

# Time
START_TIME = time.time()


def convert_color_space(img, color_space='RGB'):
    if color_space == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space != 'RGB':
        raise ValueError('Color space not valid')
    return img


def uniform_quantization(img, levels=[8, 8, 8], max_count=256):
    """
    Compute uniform quantized image indices
    :param img:
    :param levels:
    :param max_count:
    :return:
    """
    rows, cols, _ = img.shape
    img_line = img.reshape(rows * cols, -1)
    img_quantized = (np.floor(levels * img_line/max_count).astype(int) / levels)
    img_quantized_indices = (img_quantized * levels).astype(int)
    return img_quantized_indices


def features_histogram(img_quantized_indices, rows, cols, levels=[8, 8, 8]):
    """
    Compute normalized histogram of RGB/HSV image
    :param img_quantized_indices: Input quantized image in BGR color space (in this order)
    :param rows: Number of rows
    :param cols: Number of columns
    :param levels: Number of quantization levels
    :return: Normalized histogram of input image
    """
    uniques, counts = np.unique(img_quantized_indices, return_counts=True, axis=0)
    img_histogram = np.zeros(shape=levels)
    for index_3D, count in zip(uniques, counts):
        img_histogram[index_3D[0], index_3D[1], index_3D[2]] = count
    img_histogram_normalized = img_histogram.reshape(np.prod(levels)) / (rows * cols)
    return img_histogram_normalized


# MAIN
if __name__ == '__main__':
    Images_Paths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    Images_Count = len(Images_Paths)
    Features_Count = np.prod(QUANTIZE)

    for COLOR_SPACE in COLOR_SPACES:
        PARAM = str(QUANTIZE[0]) + 'x' + str(QUANTIZE[1]) + 'x' + str(QUANTIZE[2]) + COLOR_SPACE
        Histogram_Features = np.zeros((Images_Count, Features_Count))
        print(COLOR_SPACE, 'features computation ...')

        for image_index, image_path in enumerate(Images_Paths):
            Image = cv2.imread(image_path)
            Image = convert_color_space(img=Image, color_space=COLOR_SPACE)
            # Image = centering_normalization(img=Image)
            Rows, Cols, _ = Image.shape
            Image_Quantized = uniform_quantization(img=Image, levels=QUANTIZE)
            Histogram_Features[image_index] = features_histogram(
                img_quantized_indices=Image_Quantized,
                rows=Rows,
                cols=Cols,
                levels=QUANTIZE
            )
            print(image_index + 1, 'of', Images_Count, ' ', COLOR_SPACE, "--- %s seconds ---" % (time.time() - START_TIME))

        OUTPUT_FEATURES_FILE = 'features_' + ALGORITHM + PARAM + DATASET_NAME + EXTENSION
        OUTPUT_FEATURES_PATH = os.path.join(OUTPUT_FEATURES_DIRECTORY, OUTPUT_FEATURES_FILE)

        # Save features as CSV file
        np.save(OUTPUT_FEATURES_PATH, Histogram_Features)

    print("All " + ALGORITHM + " computation took", "--- %s seconds ---" % (time.time() - START_TIME))
