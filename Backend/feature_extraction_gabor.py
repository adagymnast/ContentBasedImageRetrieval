import cv2.cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils import paths
import time
import os
import pandas as pd

# Fixed constants
ALGORITHM = 'texture_gabor'

# Parameters
# DIMENSION = (200, 200)

# Fixed
LAMBDA = np.pi/4
GAMMA = 1
PHI = 0

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

# Time
START_TIME = time.time()


def show_gabor_kernels(k_size, thetas, sigma, lamda=np.pi/4, gamma=1, phi=0):
    f = plt.figure(figsize=(len(thetas) + 1, 1))
    index_subplot = 1
    for theta in thetas:
        kernel = cv2.getGaborKernel((k_size, k_size), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
        f.add_subplot(1, len(thetas), index_subplot)
        plt.imshow(kernel)
        plt.axis('off')
        f.tight_layout()
        index_subplot += 1
    plt.show(block=True)


def feature_extraction_gabor(img, k_size, sigma, thetas, lambd=np.pi/4, gamma=0.5, phi=0):
    feature_vector = np.zeros(shape=(2, len(thetas)))
    for i, theta in enumerate(thetas):
        gabor_kernel = cv2.getGaborKernel((k_size, k_size), sigma=sigma, theta=theta, lambd=lambd, gamma=gamma)
        img_filtered = cv2.filter2D(src=img, ddepth=-1, kernel=gabor_kernel).reshape(-1)
        feature_vector[0][i] = np.mean(img_filtered)
        feature_vector[1][i] = np.std(img_filtered)
    feature_vector = feature_vector.reshape(-1)
    return feature_vector


def extract_gabor(sigma, number_of_angles):

    thetas = np.pi * np.array([i / number_of_angles for i in range(number_of_angles)])
    k_size = int(2 * np.floor(3 * sigma) + 1)
    PARAM = 'S' + str(sigma) + '_' + 'A' + str(number_of_angles)

    Images_Paths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    Images_Count = len(Images_Paths)
    Features_Count = 2 * len(thetas)

    # Daisy Features
    Gabor_Features = np.zeros((Images_Count, Features_Count))

    for image_index, image_path in enumerate(Images_Paths):
        Image_Gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Gabor_Features[image_index] = feature_extraction_gabor(
            img=Image_Gray,
            k_size=k_size,
            sigma=sigma,
            thetas=thetas
        )
        print(image_index + 1, 'of', Images_Count, "--- %s seconds ---" % (time.time() - START_TIME))

    # Features
    OUTPUT_FEATURES_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Features')
    OUTPUT_FEATURES_FILE = 'features_' + ALGORITHM + PARAM + DATASET_NAME + EXTENSION
    OUTPUT_FEATURES_PATH = os.path.join(OUTPUT_FEATURES_DIRECTORY, OUTPUT_FEATURES_FILE)

    # Save features as CSV file
    np.save(OUTPUT_FEATURES_PATH, Gabor_Features)
    print("All " + ALGORITHM + " computation took", "--- %s seconds ---" % (time.time() - START_TIME))



if __name__ == '__main__':
    # show_gabor_kernels(k_size=K_SIZE, thetas=THETAS, sigma=SIGMA)
    # exit()

    NUMBER_OF_ANGLES = [4, 6, 8, 10]
    SIGMAS = [10]

    for Sigma in SIGMAS:
        for Number_Of_Angles in NUMBER_OF_ANGLES:
            extract_gabor(sigma=Sigma, number_of_angles=Number_Of_Angles)
