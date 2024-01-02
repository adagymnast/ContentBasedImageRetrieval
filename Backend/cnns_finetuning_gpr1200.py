"""
Fine-tuning of a model: to adapt for a given dataset
Model: parameter to choose
Learning rate: parameter to choose
"""

# Import libraries
import gc
import os
import cv2
import sys
import json
import time
import timm
from imutils import paths
import torch
import random
import sklearn.metrics
from PIL import Image
from pathlib import Path
from functools import partial
from contextlib import contextmanager
import numpy as np
import scipy as sp
import pandas as pd
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Normalize, Resize
from albumentations import HueSaturationValue, RandomCrop, HorizontalFlip, RandomBrightnessContrast, CenterCrop, PadIfNeeded, RandomResizedCrop, ShiftScaleRotate, Blur, JpegCompression, RandomShadow
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score
import tqdm
from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler

# Relative paths
ABSOLUTE_PATH = os.path.abspath(__file__)
FILE_DIRECTORY = os.path.dirname(ABSOLUTE_PATH)
ROOT_DIRECTORY = os.path.dirname(FILE_DIRECTORY)

# Dataset
DATASET_NAME = 'gpr1200'
INPUT_DATASET_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Datasets')
INPUT_DATASET_PATH = os.path.join(INPUT_DATASET_DIRECTORY, DATASET_NAME)

# GPU or not: 1 for GPU or 0 for not
HAVE_GPU = 0

# Number of classes: number of folders
N_CLASSES = len(next(os.walk(INPUT_DATASET_PATH))[1])

# GPU or CPU
if HAVE_GPU:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(DEVICE)

# Size of an image?
if DATASET_NAME == 'numbers':
    HEIGHT = 90
    WIDTH = 64
else:
    HEIGHT = 224
    WIDTH = 224

# Batch size
BATCH_SIZE = 20

# Number of accumulation steps
ACCUMULATION_STEPS = 1

# Number of workers
WORKERS = 0


def get_model(architecture_name, target_size, pretrained=False):
    """
    Return model
    :param architecture_name: Architecture name for timm.create_model() argument
    :param target_size: Number of classes
    :param pretrained: Boolean if pretrained model
    :return: Neural network
    """
    net = timm.create_model(architecture_name, pretrained=pretrained)
    net_cfg = net.default_cfg
    last_layer = net_cfg['classifier']
    num_features = getattr(net, last_layer).in_features
    setattr(net, last_layer, nn.Linear(num_features, target_size))
    return net


@contextmanager
def timer(name):
    """
    Timer to model logging
    :param name:
    :return:
    """
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file='train.log'):
    """
    Logger to train model
    :param log_file:
    :return:
    """
    log_format = '%(asctime)s %(levelname)s %(message)s'

    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))

    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))

    logger = getLogger('Herbarium')
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def seed_torch(seed=777):
    """
    Random seed
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class TrainDataset2Head(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['image_path'].values[idx]
        species = self.df['class_id'].values[idx]
        image = cv2.imread(file_path)

        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print('not an image', file_path)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, species


def get_transforms(*, data):
    """
    Get transformations
    :param data: 'train' or 'valid'
    :return: Transformed data depending on train/valid parameter
    """
    assert data in ('train', 'valid')

    if data == 'train':
        return Compose([
            Resize(WIDTH, HEIGHT),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=3, p=.20, border_mode=cv2.BORDER_REPLICATE),
            # JpegCompression(quality_lower=50, quality_upper=100),
            RandomShadow(),
            Blur(blur_limit=2),
            RandomBrightnessContrast(p=0.3),
            HueSaturationValue(p=0.2),
            Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(WIDTH, HEIGHT),
            Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensorV2(),
        ])


def train_and_log_model(model_name, learning_rate, epochs=3):

    # Images paths, class ids and data dictionary: initialization
    ImagesPaths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    ImagesPaths.sort()
    class_ids = dict((value, i) for i, value in enumerate(sorted(next(os.walk(INPUT_DATASET_PATH))[1])))
    data = {
        'image_path': [],
        'image_path_index': [],
        'class_id': [],
        'label': [],
        'photo': [],
    }

    # For each image: save the image path, class id, folder and photo
    for index, path in enumerate(ImagesPaths):
        photo = path.split('\\')[-1]
        folder = path.split('\\')[-2]
        class_id = class_ids[folder]
        # class_id = get_class_id(folder)

        # Save path, id, label, photo
        data['image_path'].append(path)
        data['image_path_index'].append(index)
        data['class_id'].append(class_id)
        data['label'].append(folder)
        data['photo'].append(photo)

    # Pandas dataframe from data
    metadata = pd.DataFrame.from_dict(data)

    # Split train, test pandas dataframe
    train_metadata, valid_metadata = train_test_split(metadata, test_size=0.2, random_state=1, stratify=metadata['class_id'])

    # Train and valid dataset
    train_dataset = TrainDataset2Head(train_metadata, transform=get_transforms(data='train'))
    valid_dataset = TrainDataset2Head(valid_metadata, transform=get_transforms(data='valid'))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    # Model
    model = get_model(model_name, N_CLASSES, pretrained=True)

    # model_mean = list(model.default_cfg['mean'])
    # model_std = list(model.default_cfg['std'])
    # model = nn.DataParallel(model)
    # print(model_mean)
    # print(model_std)

    with timer('Train model'):
        accumulation_steps = ACCUMULATION_STEPS
        model.to(DEVICE)

        optimizer = Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=2, verbose=True, eps=1e-6)

        criterion = nn.CrossEntropyLoss()
        best_score = 0.
        best_loss = np.inf

        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            avg_loss = 0.
            optimizer.zero_grad()
            print(train_loader)

            for i, (images, labels) in tqdm.tqdm(enumerate(train_loader)):

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                y_preds = model(images)

                # print(torch.squeeze(y_preds, 1).shape, labels.shape)
                loss = criterion(y_preds, labels)

                # Scale the loss to the mean of the accumulated batch size
                loss = loss / accumulation_steps
                loss.backward()
                if (i - 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    avg_loss += loss.item() / len(train_loader)

            model.eval()
            avg_val_loss = 0.
            preds = np.zeros((len(valid_dataset)))
            preds_raw = []

            for i, (images, labels) in enumerate(valid_loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                with torch.no_grad():
                    y_preds = model(images)

                preds[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = y_preds.argmax(1).to('cpu').numpy()
                preds_raw.extend(y_preds.to('cpu').numpy())

                loss = criterion(y_preds, labels)
                avg_val_loss += loss.item() / len(valid_loader)

            scheduler.step(avg_val_loss)

            score = f1_score(valid_metadata['class_id'], preds, average='macro')
            accuracy = accuracy_score(valid_metadata['class_id'], preds)
            test = np.array(preds_raw)

            elapsed = time.time() - start_time
            LOGGER.debug(
                f'  Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} F1: {score:.6f}  Accuracy: {accuracy:.6f} time: {elapsed:.0f}s')

            if accuracy > best_score:
                best_score = accuracy
                LOGGER.debug(f'  Epoch {epoch + 1} - Save Best Accuracy: {best_score:.6f} Model')
                torch.save(model.state_dict(), f'{OUTPUT_NAME}_best_accuracy_{DEVICE}.pth')

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                LOGGER.debug(f'  Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
                torch.save(model.state_dict(), f'{OUTPUT_NAME}_best_loss_{DEVICE}.pth')

    # torch.save(model.state_dict(), f'MobilnetV3_50E_noparallel.pth')
    torch.save(model.state_dict(), f'{OUTPUT_NAME}_{epochs}E_{DEVICE}.pth')


if __name__ == '__main__':

    MODEL_NAME = 'resnet152'

    # Number of epochs, learning rate, momentum
    EPOCHS = 200
    LEARNING_RATE = 0.00006

    LOG_EXTENSION = '.log'
    LOG_FILE = DATASET_NAME + MODEL_NAME + 'epochs' + str(EPOCHS) + 'lr' + '00006' + LOG_EXTENSION
    OUTPUT_NAME = DATASET_NAME + MODEL_NAME + 'epochs' + str(EPOCHS) + 'lr' + '00006'

    # Initialize logger and seed
    LOGGER = init_logger(LOG_FILE)
    seed_torch()

    # Train model
    train_and_log_model(
        model_name=MODEL_NAME,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS
    )
