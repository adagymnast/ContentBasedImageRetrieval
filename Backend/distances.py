# Import libraries
import numpy as np
import cv2
from imutils import paths
import time
import os
import matplotlib.pyplot as plt
import skimage
import pandas as pd
from sklearn.model_selection import train_test_split

# Import feature extraction scripts
from feature_extraction_histogram import convert_color_space, uniform_quantization, features_histogram
from feature_extraction_grid import features_grid_colors
from feature_extraction_ccv import features_ccv
from feature_extraction_glcm import features_glcm
from feature_extraction_lbp import LocalBinaryPatterns
from feature_extraction_sobel import compute_histogram, compute_sobel
from feature_extraction_robinson import compute_robinson
from feature_extraction_hog import HogDescriptor

# Evaluate dataset all parameters or select best parameters and evaluate on classes?
ON_CLASSES = False
TEST_SIZE = 0.2
MEASURES = ['precision_at_k', 'map']
MEASURE = 'map'

# Algorithm: select
ALGORITHMS = [
    'color_histogram',
    'color_grid',
    'color_ccv',
    'texture_glcm',
    'texture_lbp',
    'texture_gabor',
    'shape_sobel',
    'shape_robinson',
    'shape_hog',
    'cnn_resnet',
    'cnn_efficientnet'
]

# Algorithm parameters: select
ALGORITHM_PARAMS = {
    'color_histogram': ['8x8x8HSV', '8x8x8RGB', '8x8x4HSV', '8x8x4RGB'],
    'color_grid': ['4x4HSV', '4x4RGB', '6x6HSV', '6x6RGB', '8x8HSV', '8x8RGB', '10x10HSV', '10x10RGB'],
    'color_ccv': ['8x8x8HSV', '8x8x8RGB', '4x4x4HSV', '4x4x4RGB'],
    'texture_glcm': ['D1_A4', 'D1_A6', 'D1_A8', 'D1_A10',
                     'D2_A4', 'D2_A6', 'D2_A8', 'D2_A10',
                     'D3_A4', 'D3_A6', 'D3_A8', 'D3_A10'],
    'texture_lbp': ['P8_R1', 'P8_R2', 'P16_R2', 'P16_R3', 'P24_R3'],
    'texture_gabor': ['S1_A4', 'S1_A6', 'S1_A8', 'S1_A10',
                      'S3_A4', 'S3_A6', 'S3_A8', 'S3_A10',
                      'S10_A4', 'S10_A6', 'S10_A8', 'S10_A10'],
    'shape_sobel': ['4', '8', '16', '64', '128'],
    'shape_robinson': ['4', '8', '16', '64', '128'],
    'shape_hog': ['100x100', ''],
    'cnn_resnet': ['50', '101', '152'],
    'cnn_efficientnet': ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']
}

# Algorithm metrics: select
ALGORITHM_METRICS = {
    'color_histogram': ['l1', 'l2', 'cos'],
    'color_grid': ['l1', 'l2', 'cos'],
    'color_ccv': ['l1', 'l2', 'cos'],
    'texture_glcm': ['l1', 'l2', 'cos'],
    'texture_lbp': ['l1', 'l2', 'cos'],
    'texture_gabor': ['l1', 'l2', 'cos'],
    'shape_sobel': ['l1', 'l2', 'cos'],
    'shape_robinson': ['l1', 'l2', 'cos'],
    'shape_hog': ['l1'],
    'cnn_resnet': ['l1', 'l2', 'cos'],
    'cnn_efficientnet': ['l1', 'l2', 'cos']
}

ALGORITHM_CENTER_NORM = {
    'color_histogram': False,
    'color_grid': True,
    'color_ccv': False,
    'texture_glcm': True,
    'texture_lbp': True,
    'texture_gabor': True,
    'shape_sobel': False,
    'shape_robinson': False,
    'shape_hog': False,
    'cnn_alexnet': True,
    'cnn_vgg16': True,
    'cnn_mobilenet_v2': True,
    'cnn_resnet': True,
    'cnn_efficientnet': True
}

# Select indices of algorithm, param, metric
ALGORITHM = 'cnn_resnet'
ALGORITHM_PARAM_INDEX = 2
ALGORITHM_METRIC_INDEX = 2

ALGORITHM_PARAM = ALGORITHM_PARAMS[ALGORITHM][ALGORITHM_PARAM_INDEX]
ALGORITHM_METRIC = ALGORITHM_METRICS[ALGORITHM][ALGORITHM_METRIC_INDEX]

# If a metric is distance True, else False (if it is a similarity)
METRIC_DISTANCE = {
    'l1': True,
    'l2': True,
    'cos': True,
    'intersect': False
}

# Number of retrieved images: select
NUMBER_OF_RETRIEVED_IMAGES = 10

# Dataset: select
DATASET_NAMES = ['groundtruth', 'wang', 'patterns', 'art', 'gpr1200']
DATASET_NAME = 'gpr1200'

# Relative paths
ABSOLUTE_PATH = os.path.abspath(__file__)
FILE_DIRECTORY = os.path.dirname(ABSOLUTE_PATH)
ROOT_DIRECTORY = os.path.dirname(FILE_DIRECTORY)

# Dataset: input
EXTENSION = '.npy'
INPUT_DATASET_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Datasets')
INPUT_DATASET_PATH = os.path.join(INPUT_DATASET_DIRECTORY, DATASET_NAME)

# Features
INPUT_FEATURES_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Features')
INPUT_FEATURES_FILE = 'features_' + ALGORITHM + ALGORITHM_PARAM + DATASET_NAME + EXTENSION
INPUT_FEATURES_PATH = os.path.join(INPUT_FEATURES_DIRECTORY, INPUT_FEATURES_FILE)

# Number of relevant images for each image category
if DATASET_NAME != 'art':
    NUMBER_OF_RELEVANT_IMAGES_FOLDER = dict()
    mAP_for_each_folder = dict()
    for FOLDER in os.listdir(INPUT_DATASET_PATH):
        INPUT_DATASET_CLASS_PATH = os.path.join(INPUT_DATASET_PATH, FOLDER)
        NUMBER_OF_RELEVANT_IMAGES_FOLDER[FOLDER] = len(os.listdir(INPUT_DATASET_CLASS_PATH))
        mAP_for_each_folder[FOLDER] = 0

#########################
# _____SELECT MODE_____ #
#########################

# Select mode
MODES = ['evaluation_of_dataset', 'query_from_dataset', 'online_new_image']
MODE = 'evaluation_of_dataset'

if MODE == 'online_new_image':

    INPUT_QUERY_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Queries')
    INPUT_QUERY_PATH = os.path.join(INPUT_QUERY_DIRECTORY, DATASET_NAME)
    QUERIES_PATHS = np.array(list(paths.list_images(INPUT_QUERY_PATH)))
    QUERY_IMAGE = cv2.imread(QUERIES_PATHS[0])
    cv2.imshow('Query', QUERY_IMAGE)
    cv2.waitKey()

# Time
START_TIME = time.time()


def find_categories_and_query_indices(images_paths):
    """
    Find query indices
    :return: Query indices used for image retrieval
    """
    image_paths_sublist = np.char.split(images_paths, sep='\\')
    image_paths_np_array = np.array([image_path for image_path in image_paths_sublist])
    image_paths_classes = np.transpose(image_paths_np_array)[-2]
    uniques, counts = np.unique(image_paths_classes, return_counts=True)
    uniques_cumulative = np.cumsum(np.hstack([[0], counts]))[:-1]
    return uniques, uniques_cumulative


# Offline manner
def retrieve_image_indices(query_image_indices, features, metric, number_of_retrieved_images=10):
    """
    Compute retrieved images (number_of_retrieved_images) indices for every query image (QUERY_COUNT)
    :return: Array containing retrieved images indices (shape=(QUERY_COUNT, NUMBER_OF_RETRIEVED_IMAGES))
    """
    query_count = len(query_image_indices)
    retrieved_images_indices = np.zeros(shape=(query_count, number_of_retrieved_images))

    for i, query_index in enumerate(query_image_indices):
        query_feature = features[query_index]
        distance_values = compute_distance(
            query_feature=query_feature,
            features=features,
            metric=metric
        )
        if METRIC_DISTANCE[metric]:
            retrieved_images_indices[i] = np.argsort(distance_values)[:number_of_retrieved_images]
        else:
            retrieved_images_indices[i] = np.flip(np.argsort(distance_values))[:number_of_retrieved_images]
    return retrieved_images_indices


# Online manner
def retrieve_image_indices_online(query_features, features, metric, number_of_retrieved_images=10):
    """
    Compute retrieved images (number_of_retrieved_images) indices for every query image (QUERY_COUNT)
    :return: Array containing retrieved images indices (shape=(QUERY_COUNT, NUMBER_OF_RETRIEVED_IMAGES))
    """
    query_count = len(query_features)
    retrieved_images_indices = np.zeros(shape=(query_count, number_of_retrieved_images))

    for i, query_feature in enumerate(query_features):
        distance_values = compute_distance(
            query_feature=query_feature,
            features=features,
            metric=metric
        )
        if METRIC_DISTANCE[metric]:
            retrieved_images_indices[i] = np.argsort(distance_values)[:number_of_retrieved_images]
        else:
            retrieved_images_indices[i] = np.flip(np.argsort(distance_values))[:number_of_retrieved_images]
    return retrieved_images_indices


def compute_distance(query_feature, features, metric):
    """
    Computes distance for a given features, query index and metric
    :param query_feature: Feature of query
    :param features: Feature vectors
    :param metric: Distance measure: l1, l2, cos, intersect
    :return: Distance
    """

    if metric == 'intersect':
        images_count, features_len = features.shape
        histogram_intersections = np.zeros((images_count, features_len))
        for image_index in range(images_count):
            histogram_intersections[image_index] = np.minimum(query_feature, features[image_index])
        histogram_intersections_values = np.linalg.norm(histogram_intersections, axis=1, ord=1)  # same as sum
        return histogram_intersections_values

    elif metric == 'l1':
        return np.linalg.norm(features - query_feature, axis=1, ord=1)

    elif metric == 'l2':
        return np.linalg.norm(features - query_feature, axis=1, ord=2)

    elif metric == 'cos':
        cosine_distance = np.zeros(shape=(features.shape[0],))
        for i in range(features.shape[0]):
            a = features[i]
            b = query_feature
            cosine_distance[i] = 1 - (np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return cosine_distance

    else:
        raise


def show_retrieved_images(images_paths, retrieved_images_indices, number_of_retrieved_images, query_count, plt_size):
    """
    Shows the result of image retrieval: retrieved images for different queries
    """
    images = [
        [
            cv2.imread(images_paths[retrieved_index])
            for retrieved_index in retrieved_images_indices[query_index].astype(int)
        ]
        for query_index in range(query_count)
    ]

    f = plt.figure(figsize=plt_size)
    n_rows = query_count
    n_cols = number_of_retrieved_images
    index_subplot = 1

    for count in range(query_count):
        for image in images[count]:
            f.add_subplot(n_rows, n_cols, index_subplot)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            f.tight_layout()
            index_subplot += 1
    plt.show(block=True)
    return


# Evaluate dataset
def dataset_retrieve_and_show(images_paths, features, algorithm_metric, number_of_retrieved_images=10):

    # Choose one image per category for visualization
    categories, query_image_indices = find_categories_and_query_indices(images_paths=images_paths)
    if DATASET_NAME == 'art':
        query_image_indices = np.array([83, 4018, 15, 26, 48, 64, 69, 97, 193, 238])
    elif DATASET_NAME == 'gpr1200':
        query_image_indices = np.array([0, 2200, 2500, 2600, 4000, 7200, 7750, 8300, 10000, 11000])

    query_count = len(query_image_indices)

    # Plot parameters
    plt_size_x = 2 * number_of_retrieved_images
    plt_size_y = query_count
    plt_size = (plt_size_x, plt_size_y)

    # Retrieve image indices with a given metric: one image per category
    retrieved_images_indices = retrieve_image_indices(
        query_image_indices=query_image_indices,
        features=features,
        metric=algorithm_metric,
        number_of_retrieved_images=number_of_retrieved_images
    ).astype(int)

    # Show retrieved images: one image per category
    show_retrieved_images(
        images_paths=images_paths,
        retrieved_images_indices=retrieved_images_indices,
        number_of_retrieved_images=number_of_retrieved_images,
        query_count=query_count,
        plt_size=plt_size
    )


# Query from a dataset mode: Query retrieve, show and evaluate
def query_retrieve_and_show(images_paths, features, algorithm_metric, number_of_retrieved_images=10, query_index=0):
    """
    Retrieve a given query from a dataset
    :param images_paths: Images paths
    :param features: Features
    :param algorithm_metric:
    :param number_of_retrieved_images: Hyperparameter, specified by user
    :param query_index: Index of a query image from dataset
    :return: show retrieved images, print precision
    """

    # Plot parameters
    plt_size_x = 2 * number_of_retrieved_images
    plt_size_y = 1
    plt_size = (plt_size_x, plt_size_y)

    # Retrieve image indices with a given metric: one image per category
    retrieved_images_indices = retrieve_image_indices(
        query_image_indices=[query_index],
        features=features,
        metric=algorithm_metric,
        number_of_retrieved_images=number_of_retrieved_images
    ).astype(int)

    # Show retrieved images: one image per category
    show_retrieved_images(
        images_paths=images_paths,
        retrieved_images_indices=retrieved_images_indices,
        number_of_retrieved_images=number_of_retrieved_images,
        query_count=1,
        plt_size=plt_size
    )
    return


# Query online mode: Query retrieve and show (no evaluation)
def query_retrieve_show_online(images_paths, features, query_feature, algorithm_metric, number_of_retrieved_images=10):

    # Plot parameters
    plt_size_x = 2 * number_of_retrieved_images
    plt_size_y = 1
    plt_size = (plt_size_x, plt_size_y)

    retrieved_images_indices = retrieve_image_indices_online(
        query_features=[query_feature],
        features=features,
        metric=algorithm_metric,
        number_of_retrieved_images=number_of_retrieved_images
    ).astype(int)

    # Show retrieved images: one image per category
    show_retrieved_images(
        images_paths=images_paths,
        retrieved_images_indices=retrieved_images_indices,
        number_of_retrieved_images=number_of_retrieved_images,
        query_count=1,
        plt_size=plt_size
    )


def find_test_queries_indices(images_paths, test_size=0.2):
    """
    Finds test queries stratified by folders by given images paths and test size ratio
    :param images_paths:
    :param test_size:
    :return: Images path indices and images paths
    """
    class_ids = dict((value, i) for i, value in enumerate(sorted(next(os.walk(INPUT_DATASET_PATH))[1])))
    image_paths_with_indices = {
        'image_path': [],
        'image_path_index': [],
        'class_id': [],
    }
    for image_index, image_path in enumerate(images_paths):
        image_paths_with_indices['image_path'].append(image_path)
        image_paths_with_indices['image_path_index'].append(image_index)
        image_paths_with_indices['class_id'].append(class_ids[image_path.split('\\')[-2]])
    metadata_image_paths_with_indices = pd.DataFrame.from_dict(image_paths_with_indices)
    train_metadata, test_metadata = train_test_split(
        metadata_image_paths_with_indices, test_size=test_size, random_state=1, stratify=metadata_image_paths_with_indices['class_id'])
    return test_metadata.image_path_index.values, test_metadata.image_path.values


def dataset_evaluate_mean_average_precision(images_paths, features, algorithm_metric, n_for_average_precision, test_size, on_classes):
    """
    Evaluate whole dataset separated into categories
    :param test_size:
    :param algorithm_metric:
    :param images_paths: Images paths
    :param features: Features
    :param n_for_average_precision:
    :param on_classes: If on classes or not
    :return: Mean Average Precision
    """

    # Model evaluation: test for all query images
    # Evaluate precision and recall averages: all images from database

    images_count = len(images_paths)
    retrieved_images_indices = retrieve_image_indices(
        query_image_indices=range(images_count),
        features=features,
        metric=algorithm_metric,
        number_of_retrieved_images=n_for_average_precision
    ).astype(int)

    # Only 20 % of dataset are queries: first for loop only through test images and their indices
    query_indices, query_paths = find_test_queries_indices(images_paths=images_paths, test_size=test_size)
    average_precisions = np.zeros(shape=(len(query_paths),))

    for index, (query_index, query_path) in enumerate(zip(query_indices, query_paths)):
        query_folder = query_path.split('\\')[-2]
        query_number_of_relevant_images = NUMBER_OF_RELEVANT_IMAGES_FOLDER[query_folder]
        average_precision_sum = 0
        number_of_relevant_to_k = 0
        for retrieved_position, retrieved_image_index in enumerate(retrieved_images_indices[query_index]):
            retrieved_image_path = images_paths[retrieved_image_index]
            retrieved_image_folder = retrieved_image_path.split('\\')[-2]
            # Formula: sum rel_k * p_at_k
            if query_folder == retrieved_image_folder:
                # This case corresponds to rel_k = 1 else rel_k = 0(no need to compute p_at_k)
                number_of_relevant_to_k += 1
                p_at_k = number_of_relevant_to_k / (retrieved_position + 1)
                average_precision_sum += p_at_k

        average_precisions[index] = average_precision_sum / query_number_of_relevant_images
        # print(index, average_precisions[index])

    # mAP result
    if not on_classes:
        mean_average_precision = np.mean(average_precisions)
        return mean_average_precision

    # mAP for each class separately
    else:
        # Create dictionary of mAPs for each folder
        map_on_classes = dict()
        for folder in os.listdir(INPUT_DATASET_PATH):
            map_on_classes[folder] = 0

        for index, query_path in enumerate(query_paths):
            query_folder = query_path.split('\\')[-2]
            map_on_classes[query_folder] += average_precisions[index]
        for folder in map_on_classes.keys():
            map_on_classes[folder] = map_on_classes[folder] / (NUMBER_OF_RELEVANT_IMAGES_FOLDER[folder] * test_size)
        # These two are the same: why? why it works without dividing by test size on just mean and on classes with...?
        # print(np.mean(list(map_on_classes.values())))
        # print(np.mean(average_precisions))
        return map_on_classes


def dataset_evaluate_precision_at_k(images_paths, features, algorithm_metric, n_for_average_precision, test_size, on_classes, k):
    """
    Evaluate whole dataset separated into categories
    :param test_size:
    :param algorithm_metric:
    :param images_paths: Images paths
    :param features: Features
    :param n_for_average_precision:
    :param on_classes: If on classes or not
    :param k: k parameter for precision at k
    :return: Precision at k
    """

    # Model evaluation: test for all query images
    # Evaluate precision and recall averages: all images from database

    images_count = len(images_paths)
    retrieved_images_indices = retrieve_image_indices(
        query_image_indices=range(images_count),
        features=features,
        metric=algorithm_metric,
        number_of_retrieved_images=n_for_average_precision
    ).astype(int)

    # Only 20 % of dataset are queries: first for loop only through test images and their indices
    query_indices, query_paths = find_test_queries_indices(images_paths=images_paths, test_size=test_size)
    precisions_at_k = np.zeros(shape=(len(query_paths),))

    for index, (query_index, query_path) in enumerate(zip(query_indices, query_paths)):
        query_folder = query_path.split('\\')[-2]

        for retrieved_index in range(k):
            image_folder = images_paths[retrieved_images_indices[query_index, retrieved_index]].split('\\')[-2]
            if query_folder == image_folder:
                precisions_at_k[index] += 1
        precisions_at_k[index] = precisions_at_k[index] / k

    # P@K result
    if not on_classes:
        precision_at_k = np.mean(precisions_at_k)
        return precision_at_k

    # P@K for each class separately
    else:
        # Create dictionary of mAPs for each folder
        precision_at_k_on_classes = dict()
        for folder in os.listdir(INPUT_DATASET_PATH):
            precision_at_k_on_classes[folder] = 0

        for index, query_path in enumerate(query_paths):
            query_folder = query_path.split('\\')[-2]
            precision_at_k_on_classes[query_folder] += precisions_at_k[index]
        for folder in precision_at_k_on_classes.keys():
            precision_at_k_on_classes[folder] = precision_at_k_on_classes[folder] / (NUMBER_OF_RELEVANT_IMAGES_FOLDER[folder] * test_size)
        # These two are the same: why? why it works without dividing by test size on just mean and on classes with...?
        # print(np.mean(list(map_on_classes.values())))
        # print(np.mean(average_precisions))
        print(np.mean(precisions_at_k))
        print(np.mean(list(precision_at_k_on_classes.values())))
        return precision_at_k_on_classes


def extract_query_features(query_image, algorithm, algorithm_param):

    rows, cols, _ = query_image.shape

    if algorithm == 'color_histogram':
        color_space = algorithm_param[-3:]
        quantize_param = algorithm_param[:-3].split('x')
        quantize = np.array([int(i) for i in quantize_param])

        image = convert_color_space(img=query_image, color_space=color_space)
        image_quantized = uniform_quantization(img=image, levels=quantize)
        feature = features_histogram(
            img_quantized_indices=image_quantized,
            rows=rows,
            cols=cols,
            levels=quantize
        )
        return feature

    elif algorithm == 'color_grid':
        color_space = algorithm_param[-3:]
        quantize_param = algorithm_param[:-3].split('x')

        grid_count_x = int(quantize_param[0])
        grid_count_y = int(quantize_param[1])

        image = convert_color_space(img=query_image, color_space=color_space)
        feature = features_grid_colors(
            img=image,
            grid_count_x=grid_count_x,
            grid_count_y=grid_count_y
        )
        return feature

    elif algorithm == 'color_ccv':
        color_space = algorithm_param[-3:]
        quantize_param = algorithm_param[:-3].split('x')
        quantize = int(quantize_param[0])

        image = convert_color_space(img=query_image, color_space=color_space)
        image = cv2.resize(image, (50, 50))
        feature = features_ccv(img=image, quantize=quantize)
        return feature

    elif algorithm == 'texture_glcm':
        distances_raw, angles_raw = algorithm_param.split('_')
        distances_str = distances_raw[1:].split('x')
        distances = [int(d) for d in distances_str]
        angles_count = int(angles_raw[1:])
        angles = np.pi * np.array([i * 1/angles_count for i in range(angles_count)])

        image_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        glcm_matrix = skimage.feature.graycomatrix(
            image=image_gray,
            distances=distances,
            angles=angles,
            levels=image_gray.max()+1,
            normed=False,
            symmetric=False
        )
        feature = features_glcm(matrix_coocurrence=glcm_matrix)
        return feature

    elif algorithm == 'texture_lbp':
        points_raw, radius_raw = algorithm_param.split('_')
        points_count = int(points_raw[1:])
        radius = int(radius_raw[1:])

        image_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        feature = LocalBinaryPatterns(points_count, radius).describe(image_gray)
        return feature

    elif algorithm == 'shape_sobel':

        quantize = int(algorithm_param)

        image_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        gradient_x, gradient_y = compute_sobel(image_gray)
        edge_histogram_x = compute_histogram(gradient_x, quantize)
        edge_histogram_y = compute_histogram(gradient_y, quantize)
        feature = np.concatenate((edge_histogram_x, edge_histogram_y))
        return feature

    elif algorithm == 'shape_robinson':

        quantize = int(algorithm_param)
        feature = np.zeros((8, quantize))

        image_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        images_filtered = compute_robinson(image_gray)

        for i, image_filtered in enumerate(images_filtered):
            feature[i] = compute_histogram(image_filtered, quantize)
        feature = feature.reshape(-1)
        return feature

    elif algorithm == 'shape_hog':

        dimension_x, dimension_y = algorithm_param.split('x')
        dimension_x, dimension_y = int(dimension_x), int(dimension_y)
        dimension = (dimension_x, dimension_y)

        image_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        image_gray_resized = cv2.resize(image_gray, dimension)
        HOG = HogDescriptor(image_gray_resized, cell_size=8, bin_size=8)
        feature, _ = HOG.extract()
        return feature

    else:
        raise 'Not known algorithm'


def show_evaluation_results(precision):
    print('{:<30}{:<40}'.format('Algorithm:', ALGORITHM))
    print('{:<30}{:<40}'.format('Algorithm parameters:', ALGORITHM_PARAM))
    print('{:<30}{:<40}'.format('Metric:', ALGORITHM_METRIC))
    if precision != -1: # if evaluable: -1 means not evaluable
        print()
        print('{:<30}{:<40}'.format('Mean average precision:', precision))


def center_data(data):
    data -= data.mean(axis=0)
    return data


def norm_data(data):
    data /= np.linalg.norm(data, axis=1)[:, np.newaxis]
    return data


# MAIN
if __name__ == '__main__':
    Images_Paths = np.array(list(paths.list_images(INPUT_DATASET_PATH)))
    Images_Count = len(Images_Paths)
    Features = np.load(INPUT_FEATURES_PATH)
    # Features = norm_data(center_data(Features))
    # if ALGORITHM_CENTER_NORM[ALGORITHM]:
    #    Features = norm_data(center_data(Features))

    if MODE == 'evaluation_of_dataset':

        # dataset_retrieve_and_show(
        #     images_paths=Images_Paths,
        #     features=Features,
        #     algorithm_metric=ALGORITHM_METRIC,
        #     number_of_retrieved_images=NUMBER_OF_RETRIEVED_IMAGES,
        # )

        Precision = dataset_evaluate_mean_average_precision(
            images_paths=Images_Paths,
            features=Features,
            algorithm_metric=ALGORITHM_METRIC,
            n_for_average_precision=Images_Count,
            test_size=TEST_SIZE,
            on_classes=ON_CLASSES
        )

        show_evaluation_results(Precision)
        print("--- %s seconds ---" % (time.time() - START_TIME))

    elif MODE == 'query_from_dataset':

        QUERY_INDICES = [8]
        for QUERY_INDEX in QUERY_INDICES:
            query_retrieve_and_show(
                images_paths=Images_Paths,
                features=Features,
                algorithm_metric=ALGORITHM_METRIC,
                number_of_retrieved_images=NUMBER_OF_RETRIEVED_IMAGES,
                query_index=QUERY_INDEX,
            )

        show_evaluation_results(precision=-1)

    elif MODE == 'online_new_image':

        Query_Feature = extract_query_features(
            query_image=QUERY_IMAGE,
            algorithm=ALGORITHM,
            algorithm_param=ALGORITHM_PARAM,
        )

        query_retrieve_show_online(
            images_paths=Images_Paths,
            features=Features,
            algorithm_metric=ALGORITHM_METRIC,
            number_of_retrieved_images=NUMBER_OF_RETRIEVED_IMAGES,
            query_feature=Query_Feature,
        )

        # Not evaluable: -1 means not evaluable, just print the algorithms, params and metrics
        show_evaluation_results(precision=-1)
