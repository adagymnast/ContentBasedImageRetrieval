# Import libraries
import numpy as np
import cv2
from imutils import paths
import time
import os
import skimage

# Fixed constants
ALGORITHM = 'texture_glcm'

# Datasets
DATASETS = ['groundtruth', 'wang', 'art']
DATASET_NAME = 'patterns'
EXTENSION = '.npy'

# Relative paths
ABSOLUTE_PATH = os.path.abspath(__file__)
FILE_DIRECTORY = os.path.dirname(ABSOLUTE_PATH)
ROOT_DIRECTORY = os.path.dirname(FILE_DIRECTORY)

# Dataset
EXTENSION = '.npy'
INPUT_DATASET_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Datasets')
INPUT_DATASET_PATH = os.path.join(INPUT_DATASET_DIRECTORY, DATASET_NAME)

# Time
START_TIME = time.time()


def features_glcm(matrix_coocurrence):
    """Computes a set of 5 GLCM features"""

    asm = skimage.feature.graycoprops(matrix_coocurrence, 'ASM')
    contrast = skimage.feature.graycoprops(matrix_coocurrence, 'contrast')
    dissimilarity = skimage.feature.graycoprops(matrix_coocurrence, 'dissimilarity')
    homogeneity = skimage.feature.graycoprops(matrix_coocurrence, 'homogeneity')
    correlation = skimage.feature.graycoprops(matrix_coocurrence, 'correlation')

    features = np.array([contrast, dissimilarity, homogeneity, correlation, asm]).reshape(-1)

    return features


def extract_glcm(distances_count, angles_count):
    Images_Paths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    Images_Count = len(Images_Paths)

    # Features initialization
    Features_Count = 5 * distances_count * angles_count
    GLCM_Features = np.zeros((Images_Count, Features_Count))

    for image_index, image_path in enumerate(Images_Paths):
        Image = skimage.io.imread(image_path)
        Image_Gray = skimage.img_as_ubyte(skimage.color.rgb2gray(Image))
        Rows, Cols, _ = Image.shape

        # Quantize
        # Bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
        # Indices = np.digitize(Image_Gray, Bins)

        distances = np.array([(i + 1) for i in range(distances_count)])
        angles = np.pi * np.array([i / angles_count for i in range(angles_count)])

        # Compute GLCM matrix
        GLCM_Matrix = skimage.feature.graycomatrix(
            image=Image_Gray,
            distances=distances,
            angles=angles,
            levels=Image_Gray.max()+1,
            normed=False,
            symmetric=False
        )

        # Compute features
        GLCM_Features[image_index] = features_glcm(matrix_coocurrence=GLCM_Matrix)
        print(image_index + 1, 'of', Images_Count, "--- %s seconds ---" % (time.time() - START_TIME))

    DISTANCES_PARAM = str(distances_count)
    ANGLES_PARAM = str(angles_count)
    PARAM = 'D' + DISTANCES_PARAM + '_' + 'A' + ANGLES_PARAM

    # Features
    OUTPUT_FEATURES_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Features')
    OUTPUT_FEATURES_FILE = 'features_' + ALGORITHM + PARAM + DATASET_NAME + EXTENSION
    OUTPUT_FEATURES_PATH = os.path.join(OUTPUT_FEATURES_DIRECTORY, OUTPUT_FEATURES_FILE)

    # Save features as CSV file
    np.save(OUTPUT_FEATURES_PATH, GLCM_Features)
    print("All " + ALGORITHM + " computation took", "--- %s seconds ---" % (time.time() - START_TIME))


if __name__ == '__main__':
    Distances_Counts = [1, 2, 3]
    Angles_Counts = [4, 6, 8, 10]

    for Distances_Count in Distances_Counts:
        for Angles_Count in Angles_Counts:
            print(Distances_Count, Angles_Count)
            extract_glcm(distances_count=Distances_Count, angles_count=Angles_Count)
