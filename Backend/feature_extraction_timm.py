import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from tqdm import *
import time
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
# from GPR1200 import GPR1200


def get_sorted_distances(features_DB, features_Q=None, k=None):
    """
    Computes the cosine similarity of up to two sets of embeddings and returns
    similarities and distances of k nearest neighbours

    Parameters
    ----------
    features_DB : array-like, shape = [n_samples, dimensionality]
        Database feature vectors
    features_Q : array-like, shape = [n_samples, dimensionality]
        Query feature vectors. If this parameter is not given, the database fv´s are used as querries
    k : int
        k nearest neighbours
    """

    sorted_distances, indices = [], []

    if type(features_Q) == type(None):
        features_Q = features_DB

    if k is None:
        k = len(features_DB)

    f_db_t = features_DB.T
    for f in features_Q:
        sims = f.dot(f_db_t)

        sorted_indx = np.argsort(sims)[::-1]

        indices.append(sorted_indx[:k])
        sorted_distances.append(sims[sorted_indx[:k]])

    sorted_distances, indices = np.array(sorted_distances), np.array(indices)

    return sorted_distances, indices


def get_average_precision_score(y_true, k=None):
    """
    Average precision at rank k
    Modified to only work with sorted ground truth labels
    From: https://gist.github.com/mblondel/7337391

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Binary ground truth (True if relevant, False if irrelevant), sorted by the distances.
    k : int
        Rank.
    Returns
    -------
    average precision @k : float
    """
    if k is None:
        k = np.inf

    n_positive = np.sum(y_true.astype(np.int) == 1)

    if n_positive == 0:
        # early return in cases where no positives are among the ranks
        return 0

    y_true = y_true[:min(y_true.shape[0], k)].astype(np.int)

    score = 0
    n_positive_seen = 0
    pos_indices = np.where(y_true == 1)[0]

    for i in pos_indices:
        n_positive_seen += 1
        score += n_positive_seen / (i + 1.0)

    return score / n_positive


def compute_mean_average_precision(categories_DB,
                                   features_DB=None,
                                   features_Q=None,
                                   categories_Q=None,
                                   indices=None,
                                   k=None):
    """
    Performs a search for k neirest neighboors with the specified indexing method and computes the mean average precision@k

    Parameters
    ----------
    features_DB : array-like, shape = [n_samples, dimensionality]
        Database feature vectors
    features_Q : array-like, shape = [n_samples, dimensionality]
        Query feature vectors. If this parameter is not given, the database fv´s are used as querries
    categories_DB : array-like, shape = [n_samples_DB]
        Database categories
    categories_Q : array-like, shape = [n_samples_Q]
        Query categories. If this parameter is not given, the database categories are used
    indices: array-lile, shape = [n_samples_Q, n_samples_DB]
        Nearest neighbours indices
    k : int
        Mean average precision at @k value. If np.inf, this function computes the mean average precision score
    Returns
    -------
    Mean average precision @k : float
    """

    if (indices is None) & (features_DB is None):
        raise ValueError("Either indices or features_DB has to be provided ")

    if features_Q is None: features_Q = features_DB
    if categories_Q is None: categories_Q = categories_DB

    if (indices is None):
        _, indices = get_sorted_distances(features_DB, features_Q, k=k)

    aps = []
    for i in range(0, len(indices)):
        aps.append(get_average_precision_score((categories_DB[indices[i]] == categories_Q[i]), k))

    return aps


class GPR1200:
    """GPR1200 class

    The dataset contains 12k images from 1200 diverse categories.
    """

    _base_dir = None

    _image_data = None
    _ground_truth = None

    _iterator_index = 0

    def __init__(self, base_dir):
        """
        Load the image information from the drive

        Parameters
        ----------
        base_dir : string
            GPR1200 base directory path
        """
        self._base_dir = base_dir

        gpr10x1200_cats, gpr10x1200_files = [], []

        data = sorted(os.listdir(base_dir), key=lambda a: int(os.path.basename(a).split("_")[0]))
        for file in data:
            file_path = os.path.join(base_dir, file)
            cat = os.path.basename(file).split("_")[0]
            gpr10x1200_cats.append(cat)
            gpr10x1200_files.append(file_path)

        gpr10x1200_cats, gpr10x1200_files = np.array(gpr10x1200_cats), np.array(gpr10x1200_files)

        # sorted_indx = np.argsort(ur10x1000_files)
        self._image_files = gpr10x1200_files  # [sorted_indx]
        self._image_categories = gpr10x1200_cats  # [sorted_indx]

    @staticmethod
    def __name__():
        """
        Name of the  dataset
        """
        return "GPR1200"

    def __str__(self):
        """
        Readable string representation
        """
        return "" + self.__name__() + "(" + str(self.__len__()) + ") in " + self.base_dir

    def __len__(self):
        """
        Amount of elements
        """
        return len(self._image_data)

    @property
    def base_dir(self):
        """
        Path to the base directory

        Returns
        -------
        path : str
            Path to the base directory
        """
        return self._base_dir

    @property
    def image_dir(self):
        """
        Path to the image directory

        Returns
        -------
        path : str
            Path to the image directory
        """
        return self._base_dir + "images/"

    @property
    def image_files(self):
        """
        List of image files. The order of the list is important for other methods.

        Returns
        -------
        file_list : list(str)
            List of file names
        """
        return self._image_files

    def evaluate(self, features=None, indices=None, compute_partial=False, float_n=4):
        """
        Compute the mean average precision of each part of this combined data set.
        Providing just the 'features' will assume the manhatten distance between all images will be computed
        before calculating the mean average precision. This metric can
        be changed with any scikit learn 'distance_metric'.


        Parameters
        ----------
        features : ndarray
            matrix representing the embeddings of all the images in the dataset
        indices: array-lile, shape = [n_samples_Q, n_samples_DB]
            Nearest neighbours indices
        """

        cats = self._image_categories

        if (indices is None) & (features is None):
            raise ValueError("Either indices or features_DB has to be provided ")

        if indices is None:
            aps = compute_mean_average_precision(cats, features_DB=features)
        if features is None:
            aps = compute_mean_average_precision(cats, indices=indices)

        all_map = np.round(np.mean(aps), decimals=float_n)

        if compute_partial:
            cl_map = np.round(np.mean(aps[:2000]), decimals=float_n)
            iNat_map = np.round(np.mean(aps[2000:4000]), decimals=float_n)
            sketch_map = np.round(np.mean(aps[4000:6000]), decimals=float_n)
            instre_map = np.round(np.mean(aps[6000:8000]), decimals=float_n)
            sop_map = np.round(np.mean(aps[8000:10000]), decimals=float_n)
            faces_map = np.round(np.mean(aps[10000:]), decimals=float_n)

            return all_map, cl_map, iNat_map, sketch_map, instre_map, sop_map, faces_map

        return all_map


class TestDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, file_paths, transform):
        'Initialization'
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_paths)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Preprocessing that will be run on each individual test image
        with open(self.file_paths[index], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)
        return img


if __name__ == '__main__':
    """ 
    # Original model list
    model_list = [
                 "resnetv2_101x1_bitm",
                 "resnetv2_101x1_bitm_in21k",
                 "resnetv2_101x3_bitm",
                 "resnetv2_101x3_bitm_in21k",
                 "tf_efficientnetv2_l",
                 "tf_efficientnetv2_l_in21ft1k",
                 "tf_efficientnetv2_l_in21k",
                 "vit_base_patch16_224",
                 "vit_base_patch16_224_in21k",
                 "vit_large_patch16_224",
                 "vit_large_patch16_224_in21k",
                 "deit_base_patch16_224",
                 "deit_base_distilled_patch16_224",
                 "swin_base_patch4_window7_224",
                 "swin_base_patch4_window7_224_in22k",
                 "swin_large_patch4_window7_224",
                 "swin_large_patch4_window7_224_in22k"
                ]
    """

    model_list = [
        'vgg11',
        'vgg11_bn',
        'mobilenetv2_050',
        'mobilenetv2_100',
        'resnet50',
        'resnet101',
        'resnet152',
        "resnetv2_101x1_bitm",
        "resnetv2_101x1_bitm_in21k",
        "resnetv2_101x3_bitm",
        "resnetv2_101x3_bitm_in21k",
        'tf_efficientnet_b0',
        'tf_efficientnet_b1',
        'tf_efficientnet_b7',
        'tf_efficientnetv2_b0',
        'tf_efficientnetv2_b1',
        'tf_efficientnetv2_b3',
        "tf_efficientnetv2_l",
        "tf_efficientnetv2_l_in21ft1k",
        "tf_efficientnetv2_l_in21k",
        "swin_base_patch4_window7_224",
        "swin_base_patch4_window7_224_in22k",
        "swin_large_patch4_window7_224",
        "swin_large_patch4_window7_224_in22k"
    ]

    # Relative paths
    ABSOLUTE_PATH = os.path.abspath(__file__)
    FILE_DIRECTORY = os.path.dirname(ABSOLUTE_PATH)
    ROOT_DIRECTORY = os.path.dirname(FILE_DIRECTORY)

    # Dataset
    # DATASET_NAME = 'gpr1200_nofolders'
    DATASET_NAME = 'wang_nofolders'
    INPUT_DATASET_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Datasets')
    INPUT_DATASET_PATH = os.path.join(INPUT_DATASET_DIRECTORY, DATASET_NAME)
    # INPUT_DATASET_PATH = "../Datasets/gpr1200_nofolders"

    # GPR1200_dataset = GPR1200("../input/gpr1200-dataset/images")
    GPR1200_dataset = GPR1200(INPUT_DATASET_PATH)
    image_filepaths = GPR1200_dataset.image_files

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    for m_name in model_list:

        print(m_name)

        # create models and their respective preprocessing chain
        bb_model = timm.create_model(m_name, pretrained=True)
        data_config = resolve_data_config({}, model=bb_model)
        transform = create_transform(**data_config)

        bb_model.to(device)
        bb_model.eval()

        # dataloader parameters
        batch_size = 32
        params = {'batch_size': batch_size,
                  'shuffle': False,
                  'num_workers': 6}

        gpr1200_loader = torch.utils.data.DataLoader(TestDataset(file_paths=image_filepaths, transform=transform), **params)

        # some additional info
        time_start = time.time()
        fv_list = []

        pbar = tqdm(enumerate(gpr1200_loader), position=0, leave=True, total=(int(len(image_filepaths) / batch_size)))

        with torch.set_grad_enabled(False):
            for i, local_batch in pbar:

                local_batch = local_batch.to(device)
                fv = bb_model.forward_features(local_batch)

                if type(fv) == type((1, 2)):
                    fv = fv[1]
                if len(fv.size()) > 2:
                    fv = fv.mean(dim=[2, 3])

                fv = fv / torch.norm(fv, dim=-1, keepdim=True)

                fv_list += list(fv.cpu().numpy())
                pbar.update()

            print(fv.shape)

        # display some additional info
        fv_list = np.array(fv_list).astype(float)
        print("---------name: {} -- dim: {}---------".format(m_name, fv_list.shape))
        time_needed = np.round((time.time() - time_start) / len(image_filepaths) * 1000, 2)
        dim = fv_list.shape[-1]
        input_size = data_config["input_size"]

        # run this line to evaluate dataset embeddings
        gpr, lm, iNat, ims, instre, sop, faces = GPR1200_dataset.evaluate(fv_list, compute_partial=True)
        print("GPR1200 mAP: {}".format(gpr))
        print("Landmarks: {}, IMSketch: {}, iNat: {}, Instre: {}, SOP: {}, faces: {}".format(lm, ims, iNat, instre, sop,
                                                                                             faces))
        print()

        del bb_model
