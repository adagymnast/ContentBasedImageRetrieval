# Import libraries
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torch.nn as nn
import numpy as np
import cv2 as cv
from PIL import Image
import os
import time
from imutils import paths

# Fixed constants
ALGORITHM = 'cnn_efficientnet'

# Set batch size and number of workers
BATCH_SIZE = 64
NUM_WORKERS = 2

# Datasets
# DATASETS = ['groundtruth', 'wang', 'art']
DATASETS =['art']
EXTENSION = '.npy'

# Relative paths
ABSOLUTE_PATH = os.path.abspath(__file__)
FILE_DIRECTORY = os.path.dirname(ABSOLUTE_PATH)
ROOT_DIRECTORY = os.path.dirname(FILE_DIRECTORY)

# Dataset
EXTENSION = '.npy'
INPUT_DATASET_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Datasets')

# Features
OUTPUT_FEATURES_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Features')

# Time
START_TIME = time.time()


class ImageData(Dataset):
    def __init__(self, img=[], transform=None):
        self.transform = transform
        self.img = img
        self.dataset_N = 1

    def __len__(self):
        return self.dataset_N

    def __getitem__(self, idx):
        img = self.img
        if self.transform is not None:
            img = self.transform(img)
        return img, ''


class FeatureExtractor(object):
    def __init__(self, model_pretrained):
        print("create instance of feature_extrator")
        self.data_loader = []
        self.model_pretrained = model_pretrained

        print(model_pretrained, " architecture selected ...")
        self.model = self.load_model()
        self.model.eval()

    def load_model(self):
        """
        Loads EfficientNet architecture model
        :return:
        """
        print("Loading", self.model_pretrained, "architecture ...")

        if self.model_pretrained == 'efficientnetb0':
            model = models.efficientnet_b0(pretrained=True)

        elif self.model_pretrained == 'efficientnetb1':
            model = models.efficientnet_b1(pretrained=True)

        elif self.model_pretrained == 'efficientnetb2':
            model = models.efficientnet_b2(pretrained=True)

        elif self.model_pretrained == 'efficientnetb3':
            model = models.efficientnet_b3(pretrained=True)

        elif self.model_pretrained == 'efficientnetb4':
            model = models.efficientnet_b4(pretrained=True)

        elif self.model_pretrained == 'efficientnetb5':
            model = models.efficientnet_b5(pretrained=True)

        elif self.model_pretrained == 'efficientnetb6':
            model = models.efficientnet_b6(pretrained=True)

        elif self.model_pretrained == 'efficientnetb7':
            model = models.efficientnet_b7(pretrained=True)

        else:
            raise 'Select EfficientNet model from models'

        print('Loaded architecture')
        for param in model.parameters():
            param.requires_grad = False
        model = nn.Sequential(*list(model.children())[:-1])
        return model

    def get_features_numpy(self):
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.data_loader):
                # 1 element
                inputs = data[0]
                tensors = inputs
                print(f'Processing image feature...')
                features_tensor = self.model(tensors)
                features_tensor = torch.flatten(features_tensor, start_dim=1)
                features_np = features_tensor.detach().cpu().numpy()
        return features_np


def get_image_features_numpy(img, retrieval_data, retrieval_extractor):
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_RGB)
    retrieval_data.img = img_pil
    img_loader = DataLoader(retrieval_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    retrieval_extractor.data_loader = img_loader
    features_np = retrieval_extractor.get_features_numpy()
    return features_np


def decorate_img(img, l2norm, img_height=372):
    """
    Decorates an image with a computed image distance
    :param img:
    :param l2norm:
    :param img_height:
    :return:
    """
    scale = img_height / img.shape[0]
    width = int(img.shape[1] * scale)
    new_size = (width, img_height)
    img = cv.resize(img, new_size, interpolation=cv.INTER_CUBIC)

    font = cv.FONT_HERSHEY_SIMPLEX
    x, y, w, h = 0, 0, 200, 50
    img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
    img = cv.putText(img, f'dis = {l2norm:.4f}', (x + int(w / 10), y + int(h / 2)),
                     font, 0.7, (255, 255, 255), 2)
    return img


def visualize_comparison(imgs, l2norms):
    img_decorated = [decorate_img(img, l2norm) for img, l2norm in zip(imgs, l2norms)]
    img_final = np.concatenate(img_decorated, axis=1)
    return img_final


def get_features_folder(folder, retrieval_data, retrieval_extractor):
    """
    Finds all images in a given folder
    :param folder:
    :return:
    """
    images_paths = np.array(list(paths.list_images(folder)))
    images = [cv.imread(img) for img in images_paths]
    features_all = np.concatenate([get_image_features_numpy(img, retrieval_data, retrieval_extractor) for img in images])
    return features_all, images_paths, images


def compare_features(file, imgs, features_all):
    img = cv.imread(file)
    features_img = get_image_features_numpy(img)
    diff = features_all - features_img
    l2_norm = np.linalg.norm(diff, axis=1)
    ind_sorted = l2_norm.argsort()
    img_out = visualize_comparison(list(imgs[i] for i in ind_sorted), l2_norm[ind_sorted])

    return img_out


# Feature extraction efficientnet
def feature_extraction_efficientnet(dataset_name, model_pretrained, output_features_path, param):

    dataset_path = os.path.join(INPUT_DATASET_DIRECTORY, dataset_name)
    # Transform operation
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Retrieval data, retrieval extractor instance
    retrieval_data = ImageData(transform=transformation)

    # Models
    retrieval_extractor = FeatureExtractor(model_pretrained=model_pretrained)

    # Extract features
    EfficientNet_Features, Files, Images = get_features_folder(
        folder=dataset_path,
        retrieval_data=retrieval_data,
        retrieval_extractor=retrieval_extractor
    )

    # Save features as CSV file
    np.save(output_features_path, EfficientNet_Features)
    print("All " + ALGORITHM + param + " computation took", "--- %s seconds ---" % (time.time() - START_TIME))


if __name__ == '__main__':

    # Parameters: architectures of EfficientNet network
    PARAMS = ['b3']
    MODELS = ['efficientnet' + param for param in PARAMS]

    for DATASET_NAME in DATASETS:
        for PARAM, MODEL in zip(PARAMS, MODELS):

            print('Run of:', DATASET_NAME, MODEL)

            OUTPUT_FEATURES_FILE = 'features_' + ALGORITHM + PARAM + DATASET_NAME + EXTENSION
            OUTPUT_FEATURES_PATH = os.path.join(OUTPUT_FEATURES_DIRECTORY, OUTPUT_FEATURES_FILE)

            feature_extraction_efficientnet(
                dataset_name=DATASET_NAME,
                model_pretrained=MODEL,
                output_features_path=OUTPUT_FEATURES_PATH,
                param=PARAM
            )
