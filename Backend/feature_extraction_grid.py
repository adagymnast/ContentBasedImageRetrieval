# Import libraries
import numpy as np
import cv2
import math
from imutils import paths
import time
import os

# Fixed constants
ALGORITHM = 'color_grid'

# Parameters
GRID_COUNT_X, GRID_COUNT_Y = 4, 4

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


def features_grid_colors(img, max_count=256, grid_count_x=4, grid_count_y=4):
    """
    Compute average color for every block of the grid
    :param grid_count_y: Number of grid blocks in y
    :param grid_count_x: Number of grid blocks in x
    :param max_count: Maximum count of range (usually 256)
    :param img: Input image in BGR color space (in this order)
    :param color_space: Color space (RGB or HSV)
    :return: Average colors for all blocks of the grid (shape=GRID_COUNT_X * GRID_COUNT_Y * 3)
    """
    rows_y, cols_x, channels = img.shape
    img_normalized = img / max_count

    # Compute grid size
    grid_size_x = math.floor(cols_x / grid_count_x)
    grid_size_y = math.floor(rows_y / grid_count_y)

    img_averages = np.zeros(shape=(grid_count_x, grid_count_y, 3))

    for grid_x in range(grid_count_x):
        for grid_y in range(grid_count_y):
            img_region = img_normalized[
                               (grid_y * grid_size_y):((grid_y + 1) * grid_size_y),
                               (grid_x * grid_size_x):((grid_x + 1) * grid_size_x)
                               ]
            img_region_list = img_region.reshape(img_region.shape[0] * img_region.shape[1], img_region.shape[2])
            img_averages[grid_x][grid_y][:3] = np.mean(img_region_list, axis=0)

    return img_averages.reshape(grid_count_x * grid_count_y * 3)


# MAIN
if __name__ == '__main__':
    Images_Paths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    Images_Count = len(Images_Paths)

    # Parameters
    COLOR_SPACES = ['RGB', 'HSV']
    for COLOR_SPACE in COLOR_SPACES:
        PARAM = str(GRID_COUNT_X) + 'x' + str(GRID_COUNT_Y) + COLOR_SPACE
        Grid_Features = np.zeros((Images_Count, GRID_COUNT_X * GRID_COUNT_Y * 3))
        print(COLOR_SPACE, 'features computation ...')

        for image_index, image_path in enumerate(Images_Paths):
            Image = cv2.imread(image_path)
            Image = convert_color_space(img=Image, color_space=COLOR_SPACE)
            Grid_Features[image_index] = features_grid_colors(
                img=Image, grid_count_x=GRID_COUNT_X, grid_count_y=GRID_COUNT_Y
            )

        OUTPUT_FEATURES_FILE = 'features_' + ALGORITHM + PARAM + DATASET_NAME + EXTENSION
        OUTPUT_FEATURES_PATH = os.path.join(OUTPUT_FEATURES_DIRECTORY, OUTPUT_FEATURES_FILE)

        # Save features as CSV file
        # np.savetxt(OUTPUT_FEATURES_PATH, Grid_Features, delimiter=',')
        np.save(OUTPUT_FEATURES_PATH, Grid_Features)

    print("All " + ALGORITHM + " computation took", "--- %s seconds ---" % (time.time() - START_TIME))
