import os
import random
import numpy as np
import cv2
from imutils import paths
import json
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from sklearn.manifold import TSNE

ALGORITHM = 'cnn_efficientnet'
ALGORITHM_PARAM = 'b0'

# Relative paths
ABSOLUTE_PATH = os.path.abspath(__file__)
FILE_DIRECTORY = os.path.dirname(ABSOLUTE_PATH)
ROOT_DIRECTORY = os.path.dirname(FILE_DIRECTORY)

# Dataset: input
DATASET_NAME = 'art'
EXTENSION = '.npy'
INPUT_DATASET_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Datasets')
INPUT_DATASET_PATH = os.path.join(INPUT_DATASET_DIRECTORY, DATASET_NAME)

# Features
INPUT_FEATURES_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Features')
INPUT_FEATURES_FILE = 'features_' + ALGORITHM + ALGORITHM_PARAM + DATASET_NAME + EXTENSION
# INPUT_FEATURES_FILE = 'features_cnn_finetuned_resnet152_0001patterns' + EXTENSION
INPUT_FEATURES_PATH = os.path.join(INPUT_FEATURES_DIRECTORY, INPUT_FEATURES_FILE)


if __name__ == '__main__':
    Images_Paths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    Images_Count = len(Images_Paths)
    Features = np.load(INPUT_FEATURES_PATH)
    Images = [cv2.imread(Images_Paths[index]) for index in range(Images_Count)]

    for image, f in list(zip(Images, Features))[0:5]:
        print("image: %s, features: %0.2f,%0.2f,%0.2f,%0.2f... " % (image.shape, f[0], f[1], f[2], f[3]))

    tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2, n_iter=1000, n_iter_without_progress=1000).fit_transform(Features)
    tx, ty = tsne[:, 0], tsne[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 4000
    height = 3000
    max_dim = 120

    full_image = Image.new('RGBA', (width, height))
    for img, x, y in zip(Images_Paths, tx, ty):
        tile = Image.open(img)
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))

    plt.figure(figsize=(16, 12))
    plt.axis('off')
    plt.imshow(full_image)
    plt.show()
