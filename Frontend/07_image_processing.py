from flask import Flask, request, render_template
import cv2 as cv
import numpy as np
import datetime
import os
from imutils import paths
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torch.nn as nn
from PIL import Image
import time

app = Flask(__name__)

# Relative paths
ABSOLUTE_PATH = os.path.abspath(__file__)
FILE_DIRECTORY = os.path.dirname(ABSOLUTE_PATH)
ROOT_DIRECTORY = os.path.dirname(FILE_DIRECTORY)

# Dataset: select
DATASET_NAME = 'art'
INPUT_DATASET_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Datasets')
INPUT_DATASET_PATH = os.path.join(INPUT_DATASET_DIRECTORY, DATASET_NAME)

# Features
EXTENSION = '.npy'
FEATURES_NAME = 'features_cnn_efficientnetb3' + DATASET_NAME
INPUT_FEATURES_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Features')
INPUT_FEATURES_FILE = FEATURES_NAME + EXTENSION
INPUT_FEATURES_PATH = os.path.join(INPUT_FEATURES_DIRECTORY, INPUT_FEATURES_FILE)

# Parameters
BATCH_SIZE = 64
NUM_WORKERS = 2


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
        model = models.efficientnet_b3(pretrained=True)
        print('Loaded model architecture')
        for param in model.parameters():
            param.requires_grad = False
        model = nn.Sequential(*list(model.children())[:-1])
        return model

    def get_features_numpy(self):
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.data_loader):
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


# Feature extraction efficientnet
def feature_extraction_efficientnet(img, model_pretrained):
    # Transform operation
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Retrieval data, retrieval extractor instance
    retrieval_data = ImageData(transform=transformation)
    retrieval_extractor = FeatureExtractor(model_pretrained=model_pretrained)
    feature = get_image_features_numpy(img, retrieval_data, retrieval_extractor)
    return feature


def compute_cosine_distance(query_feature, features):
    """
    Computes distance for a given features, query index and metric
    :param query_feature: Feature of query
    :param features: Feature vectors
    :return: Distance
    """
    cosine_distance = np.zeros(shape=(features.shape[0],))
    for i in range(features.shape[0]):
        a = features[i]
        b = query_feature
        cosine_distance[i] = 1 - (np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return cosine_distance


@app.route("/")
def index():
    return render_template("02.html")


@app.route('/rgb2gray', methods=['POST'])
def sending_image():

    # Load images paths and image features
    images_paths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    features = np.load(INPUT_FEATURES_PATH)

    # File
    file = request.files['file']
    print("Posted file - SENDING: {}".format(file))

    # Get image and set dimension
    img = cv.imdecode(np.frombuffer(file.read(), np.uint8), cv.IMREAD_UNCHANGED)
    width = 500
    scale_percent = img.shape[1] / width
    height = int(img.shape[0] / scale_percent)
    dim = (width, height)
    img = cv.resize(img, dim, interpolation=cv.INTER_CUBIC)

    # Convert image to 1536-dimensional vector using EfficientNetb3 architecture
    number_of_retrieved_images=5
    model_pretrained='efficientnetb3'
    query_feature = feature_extraction_efficientnet(img=img, model_pretrained=model_pretrained)
    distance_values = compute_cosine_distance(query_feature=query_feature, features=features)
    retrieved_images_indices = np.argsort(distance_values)[:number_of_retrieved_images].astype(int)
    retrieved_images = [cv.imread(images_paths[retrieved_index]) for retrieved_index in retrieved_images_indices]

    # Gray image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Write image
    extension = file.filename.split(".")[-1]

    retrieved_image_1 = retrieved_images[0]
    retrieved_image_2 = retrieved_images[1]
    retrieved_image_3 = retrieved_images[2]
    retrieved_image_4 = retrieved_images[3]
    retrieved_image_5 = retrieved_images[4]

    filename = f'static/images/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f.") + extension}'
    filename_1 = f'static/images/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f_retrieved1.") + extension}'
    filename_2 = f'static/images/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f_retrieved2.") + extension}'
    filename_3 = f'static/images/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f_retrieved3.") + extension}'
    filename_4 = f'static/images/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f_retrieved4.") + extension}'
    filename_5 = f'static/images/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f_retrieved5.") + extension}'

    cv.imwrite(filename, img)
    cv.imwrite(filename_1, retrieved_image_1)
    cv.imwrite(filename_2, retrieved_image_2)
    cv.imwrite(filename_3, retrieved_image_3)
    cv.imwrite(filename_4, retrieved_image_4)
    cv.imwrite(filename_5, retrieved_image_5)

    # filename_gray = f'static/images/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f_gray.") + extension}'
    # cv.imwrite(filename_gray, gray)

    # Return
    return render_template(
        "02_out.html",
        img_file_original=filename,
        img_file_retrieved_1=filename_1,
        img_file_retrieved_2=filename_2,
        img_file_retrieved_3=filename_3,
        img_file_retrieved_4=filename_4,
        img_file_retrieved_5=filename_5
    )


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
