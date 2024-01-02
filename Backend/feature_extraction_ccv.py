# Import libraries
import numpy as np
import cv2
import math
from imutils import paths
import time
import os

# Fixed constants
ALGORITHM = 'color_ccv'

# Parameters
COLOR_SPACES = ['RGB', 'HSV']
QUANTIZE = 8  # indicating QUANTIZE^3 discretized colors
CCV_LEN = QUANTIZE ** 3 * 2  # Number of discretized colors * 2 for coherent and incoherent
# IMAGE_DIMENSION = (50, 50)

# Relative paths
ABSOLUTE_PATH = os.path.abspath(__file__)
FILE_DIRECTORY = os.path.dirname(ABSOLUTE_PATH)
ROOT_DIRECTORY = os.path.dirname(FILE_DIRECTORY)

# Dataset
DATASET_NAME = 'patterns'
EXTENSION = '.npy'
# INPUT_DATASET_DIRECTORY = r"C:\Users\adela\PycharmProjects\MasterThesisProjects\Datasets"
INPUT_DATASET_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Datasets')
INPUT_DATASET_PATH = os.path.join(INPUT_DATASET_DIRECTORY, DATASET_NAME)

# Features
OUTPUT_FEATURES_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Features')
# OUTPUT_FEATURES_DIR = r"C:\Users\adela\PycharmProjects\MasterThesisProjects\Features"

# Time
START_TIME = time.time()


def convert_color_space(img, color_space='RGB'):
    if color_space == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space != 'RGB':
        raise ValueError('Color space not valid')
    return img


def is_adjacent(x1, y1, x2, y2):
    """
    Returns true if (x1, y1) is adjacent to (x2, y2), and false otherwise
    """
    x_diff = abs(x1 - x2)
    y_diff = abs(y1 - y2)
    return not (x_diff == 1 and y_diff == 1) and (x_diff <= 1 and y_diff <= 1)


def find_max_cliques(img, n):
    """
    Returns a 2*n dimensional vector
    v_i, v_{i+1} describes the number of coherent and incoherent pixels respectively a given color
    :param img:
    :param n:
    :return: Color coherent vector
    """
    tau = int(img.shape[0] * img.shape[1] * 0.01)  # Classify as coherent is area is >= 1%
    ccv = [0 for _ in range(2 * n ** 3)]
    unique = np.unique(img)
    for u in unique:
        x, y = np.where(img == u)
        groups = []
        coherent, incoherent = 0, 0

        for i in range(len(x)):
            found_group = False
            for group in groups:
                if found_group:
                    break

                for coord in group:
                    xj, yj = coord
                    if is_adjacent(x[i], y[i], xj, yj):
                        found_group = True
                        group[(x[i], y[i])] = 1
                        break
            if not found_group:
                groups.append({(x[i], y[i]): 1})

        for group in groups:
            num_pixels = len(group)
            if num_pixels >= tau:
                coherent += num_pixels
            else:
                incoherent += num_pixels

        assert (coherent + incoherent == len(x))

        index = int(u)
        ccv[index * 2] = coherent
        ccv[index * 2 + 1] = incoherent

    return ccv


def features_ccv(img, quantize=8):
    """
    Computes CCV features
    :param img:
    :param quantize:
    :return:
    """
    # Blur pixel slightly using avg pooling with 3x3 kernel
    blur_img = cv2.blur(img, (3, 3))
    blur_flat = blur_img.reshape(-1, 3)

    # Discretize colors
    hist, edges = np.histogramdd(blur_flat, bins=quantize)

    graph = np.zeros((img.shape[0], img.shape[1]))
    result = np.zeros(blur_img.shape)

    for i in range(0, quantize):
        for j in range(0, quantize):
            for k in range(0, quantize):
                rgb_val = [edges[0][i + 1], edges[1][j + 1], edges[2][k + 1]]
                previous_edge = [edges[0][i], edges[1][j], edges[2][k]]
                coords = ((blur_img <= rgb_val) & (blur_img >= previous_edge)).all(axis=2)
                result[coords] = rgb_val
                graph[coords] = i + j * quantize + k * quantize ** 2

    return find_max_cliques(graph, quantize)


# MAIN
if __name__ == '__main__':
    Images_Paths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    Images_Count = len(Images_Paths)

    for COLOR_SPACE in COLOR_SPACES:
        PARAM = str(QUANTIZE) + 'x' + str(QUANTIZE) + 'x' + str(QUANTIZE) + COLOR_SPACE
        CCV_Features = np.zeros((Images_Count, CCV_LEN))
        print(COLOR_SPACE, 'features computation ...')

        for image_index, image_path in enumerate(Images_Paths):
            Image = cv2.imread(image_path)
            Image = convert_color_space(img=Image, color_space=COLOR_SPACE)
            Rows, Cols, _ = Image.shape
            if Rows >= Cols:
                IMAGE_DIMENSION = (96, 64)
            else:
                IMAGE_DIMENSION = (64, 96)

            Image = cv2.resize(Image, IMAGE_DIMENSION)
            CCV_Features[image_index] = features_ccv(img=Image, quantize=QUANTIZE)
            print(image_index + 1, 'of', Images_Count, ' ', COLOR_SPACE, "--- %s seconds ---" % (time.time() - START_TIME))

        OUTPUT_FEATURES_FILE = 'features_' + ALGORITHM + PARAM + DATASET_NAME + EXTENSION
        OUTPUT_FEATURES_PATH = os.path.join(OUTPUT_FEATURES_DIRECTORY, OUTPUT_FEATURES_FILE)

        # Save features as CSV file
        np.save(OUTPUT_FEATURES_PATH, CCV_Features)
    print("All " + ALGORITHM + " computation took", "--- %s seconds ---" % (time.time() - START_TIME))
