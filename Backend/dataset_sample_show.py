import matplotlib.pyplot as plt
import cv2
import numpy as np
from imutils import paths
import os

# Relative paths
ABSOLUTE_PATH = os.path.abspath(__file__)
FILE_DIRECTORY = os.path.dirname(ABSOLUTE_PATH)
ROOT_DIRECTORY = os.path.dirname(FILE_DIRECTORY)

# Dataset: input
# DATASET_NAMES = ['art_subset', 'wang']
DATASET_NAME = 'patterns'
EXTENSION = '.npy'


if DATASET_NAME == 'wang':
    INPUT_DATASET_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Datasets')
    INPUT_DATASET_PATH = os.path.join(INPUT_DATASET_DIRECTORY, DATASET_NAME)

    images_indices = [64, 102, 250, 350, 405, 500, 652, 700, 800, 900]
    images_paths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    images = [
            cv2.imread(images_paths[image_index])
            for image_index in images_indices
    ]

    f = plt.figure(figsize=(10, 4))
    n_rows = 2
    n_cols = 5
    index_subplot = 1

    for image in images:
        f.add_subplot(n_rows, n_cols, index_subplot)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        f.tight_layout()
        index_subplot += 1
    plt.show(block=True)

elif DATASET_NAME == 'art_subset':
    INPUT_DATASET_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Queries')
    INPUT_DATASET_PATH = os.path.join(INPUT_DATASET_DIRECTORY, DATASET_NAME)

    images_paths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    images_paths.sort()
    images = [cv2.imread(images_path) for images_path in images_paths]

    f = plt.figure(figsize=(10, 4))
    n_rows = 2
    n_cols = 6
    index_subplot = 1

    for image in images:
        f.add_subplot(n_rows, n_cols, index_subplot)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        f.tight_layout()
        index_subplot += 1
    plt.show(block=True)

elif DATASET_NAME == 'patterns':
    INPUT_DATASET_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Datasets')
    INPUT_DATASET_PATH = os.path.join(INPUT_DATASET_DIRECTORY, DATASET_NAME)

    images_indices = [35, 123, 247, 360, 481, 602, 723, 840, 963, 1081]
    images_paths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    images = [
            cv2.imread(images_paths[image_index])
            for image_index in images_indices
    ]

    f = plt.figure(figsize=(10, 4))
    n_rows = 2
    n_cols = 5
    index_subplot = 1

    for image in images:
        f.add_subplot(n_rows, n_cols, index_subplot)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        f.tight_layout()
        index_subplot += 1
    plt.show(block=True)
