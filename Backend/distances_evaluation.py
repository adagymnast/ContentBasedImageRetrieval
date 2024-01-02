# Import libraries
import numpy as np
import cv2
from imutils import paths
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Evaluate dataset all parameters or select best parameters and evaluate on classes?
ON_CLASSES = True
CENTER_NORM = False
TEST_SIZE = 0.2
# MEASURES = ['precision_at_k', 'map']
MEASURE = 'map'
# CENTER_NORM = False

# Algorithm: select
ALGORITHMS = [
    # 'color_histogram',
    # 'color_grid',
    # 'color_ccv',
    # 'texture_glcm',
    # 'texture_lbp',
    # 'texture_gabor',
    # 'shape_sobel',
    # 'shape_robinson',
    # 'shape_hog',
    # 'cnn_alexnet',
    # 'cnn_vgg16',
    # 'cnn_mobilenet_v2',
    # 'cnn_resnet',
    'cnn_finetuned_resnet'
    # 'cnn_efficientnet'
]

# Algorithm parameters: select
ALGORITHM_PARAMS = {
    'color_histogram': ['8x8x8HSV', '8x8x8RGB'],
    'color_grid': ['4x4HSV', '4x4RGB', '6x6HSV', '6x6RGB', '8x8HSV', '8x8RGB', '10x10HSV', '10x10RGB'],
    'color_ccv': ['8x8x8HSV', '8x8x8RGB'],
    'texture_glcm': ['D1_A4', 'D1_A6', 'D1_A8', 'D1_A10',
                     'D2_A4', 'D2_A6', 'D2_A8', 'D2_A10',
                     'D3_A4', 'D3_A6', 'D3_A8', 'D3_A10'],
    'texture_lbp': ['P8_R1', 'P8_R2', 'P16_R2', 'P16_R3', 'P24_R3'],
    'texture_gabor': ['S1_A4', 'S1_A6', 'S1_A8', 'S1_A10',
                      'S3_A4', 'S3_A6', 'S3_A8', 'S3_A10',
                      'S10_A4', 'S10_A6', 'S10_A8', 'S10_A10'],
    'shape_sobel': ['4', '8', '16', '64', '128'],
    'shape_robinson': ['4', '8', '16', '64', '128'],
    'shape_hog': ['100x100'],
    'cnn_alexnet': [''],
    'cnn_vgg16': [''],
    'cnn_mobilenet_v2': [''],
    'cnn_resnet': ['152'],
    'cnn_finetuned_resnet' : ['152_'],
    'cnn_efficientnet': ['b0', 'b3', 'b7']
}
# 152_0001

# Algorithm metrics: select
ALGORITHM_METRICS = {
    'color_histogram': ['l1', 'l2', 'cos', 'intersect'],
    'color_grid': ['l1', 'l2', 'cos'],
    'color_ccv': ['l1', 'l2', 'cos', 'intersect'],
    'texture_glcm': ['l1', 'l2', 'cos'],
    'texture_lbp': ['l1', 'l2', 'cos'],
    'texture_gabor': ['l1', 'l2', 'cos'],
    'shape_sobel': ['l1', 'l2', 'cos', 'intersect'],
    'shape_robinson': ['l1', 'l2', 'cos', 'intersect'],
    'shape_hog': ['l1', 'l2', 'cos', 'intersect'],
    'cnn_alexnet': ['l1', 'l2', 'cos'],
    'cnn_vgg16': ['l1', 'l2', 'cos'],
    'cnn_mobilenet_v2': ['l1', 'l2', 'cos'],
    'cnn_resnet': ['l1', 'l2', 'cos'],
    'cnn_finetuned_resnet': ['cos'],
    'cnn_efficientnet': ['l1', 'l2', 'cos']
}

ALGORITHM_PARAMS_BEST = {
    'color_histogram': '8x8x8HSV',
    'color_grid': '6x6HSV',
    'color_ccv': '8x8x8RGB',
    'texture_glcm': 'D3_A4',
    'texture_lbp': 'P16_R2',
    'texture_gabor': 'S3_A8',
    'shape_sobel': '128',
    'shape_robinson': '128',
    'shape_hog': '100x100',
    'cnn_alexnet': '',
    'cnn_vgg16': '',
    'cnn_mobilenet_v2': '',
    'cnn_resnet': '152',
    'cnn_finetuned_resnet': '_131epochs',
    'cnn_efficientnet': 'b3'
}

ALGORITHM_METRICS_BEST = {
    'color_histogram': 'l1',
    'color_grid': 'l1',
    'color_ccv': 'l1',
    'texture_glcm': 'l1',
    'texture_lbp': 'l1',
    'texture_gabor': 'cos',
    'shape_sobel': 'l1',
    'shape_robinson': 'l1',
    'shape_hog': 'l1',
    'cnn_alexnet': 'cos',
    'cnn_vgg16': 'cos',
    'cnn_mobilenet_v2': 'cos',
    'cnn_resnet': 'cos',
    'cnn_finetuned_resnet': 'cos',
    'cnn_efficientnet': 'cos'
}

# If a metric is distance True, else False (if it is a similarity)
METRIC_DISTANCE = {
    'l1': True,
    'l2': True,
    'cos': True,
    'intersect': False
}

# Number of retrieved images: select
NUMBER_OF_RETRIEVED_IMAGES = 10
EXTENSION = '.npy'

# Relative paths
ABSOLUTE_PATH = os.path.abspath(__file__)
FILE_DIRECTORY = os.path.dirname(ABSOLUTE_PATH)
ROOT_DIRECTORY = os.path.dirname(FILE_DIRECTORY)

# Features
INPUT_FEATURES_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Features')

# Dataset: select
DATASET_NAME = 'patterns'
INPUT_DATASET_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'Datasets')
INPUT_DATASET_PATH = os.path.join(INPUT_DATASET_DIRECTORY, DATASET_NAME)

# Number of relevant images for each image category
NUMBER_OF_RELEVANT_IMAGES_FOLDER = dict()
for FOLDER in os.listdir(INPUT_DATASET_PATH):
    INPUT_DATASET_CLASS_PATH = os.path.join(INPUT_DATASET_PATH, FOLDER)
    NUMBER_OF_RELEVANT_IMAGES_FOLDER[FOLDER] = len(os.listdir(INPUT_DATASET_CLASS_PATH))
NUMBER_OF_FOLDERS = len(os.listdir(INPUT_DATASET_PATH))

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
        # print(np.linalg.norm(features - query_feature, axis=1, ord=1).shape)
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


def show_retrieved_images(images_paths, retrieved_images_indices, number_of_retrieved_images, query_count, plt_size, plt_path):
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
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            f.tight_layout()
            index_subplot += 1
    plt.savefig(plt_path)
    # plt.show(block=True)
    return


# Evaluate dataset
def dataset_retrieve_and_show(images_paths, features, algorithm, algorithm_param, algorithm_metric, number_of_retrieved_images=10):

    # Choose one image per category for visualization
    categories, query_image_indices = find_categories_and_query_indices(images_paths=images_paths)
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
    # Plot path
    plt_path = algorithm + '_' + algorithm_param + '_' + algorithm_metric
    show_retrieved_images(
        images_paths=images_paths,
        retrieved_images_indices=retrieved_images_indices,
        number_of_retrieved_images=number_of_retrieved_images,
        query_count=query_count,
        plt_size=plt_size,
        plt_path=plt_path,
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


def center_data(data):
    data -= data.mean(axis=0)
    return data


def norm_data(data):
    data /= np.linalg.norm(data, axis=1)[:, np.newaxis]
    return data


def image_retrieval_evaluation(input_dataset_path, algorithm, algorithm_param, algorithm_metric, number_of_retrieved_images=10, test_size=0.2, on_classes=False, measure='map'):
    """
    Only mode: evaluation of the dataset
    :param test_size:
    :param input_dataset_path:
    :param algorithm:
    :param algorithm_param:
    :param algorithm_metric:
    :param number_of_retrieved_images:
    :return:
    """

    # Dataset images paths
    Images_Paths = np.array(list(paths.list_images(input_dataset_path)))
    Images_Count = len(Images_Paths)

    # Features
    input_features_file = 'features_' + algorithm + algorithm_param + DATASET_NAME + EXTENSION
    input_features_path = os.path.join(INPUT_FEATURES_DIRECTORY, input_features_file)
    Features = np.load(input_features_path)
    if CENTER_NORM:
       Features = norm_data(center_data(Features))

    # dataset_retrieve_and_show(
    #     images_paths=Images_Paths,
    #     features=Features,
    #     algorithm=algorithm,
    #     algorithm_param=algorithm_param,
    #     algorithm_metric=algorithm_metric,
    #     number_of_retrieved_images=number_of_retrieved_images,
    # )

    if measure == 'map':
        precision = dataset_evaluate_mean_average_precision(
            images_paths=Images_Paths,
            features=Features,
            algorithm_metric=algorithm_metric,
            n_for_average_precision=Images_Count,
            test_size=test_size,
            on_classes=on_classes
        )
    elif measure == 'precision_at_k':
        precision = dataset_evaluate_precision_at_k(
            images_paths=Images_Paths,
            features=Features,
            algorithm_metric=algorithm_metric,
            n_for_average_precision=Images_Count,
            test_size=test_size,
            on_classes=on_classes,
            k=10
        )

    return precision


if __name__ == '__main__':

    # Evaluation of all parameters
    if not ON_CLASSES:
        for Algorithm in ALGORITHMS:
            df_precision_results = pd.DataFrame(ALGORITHM_PARAMS[Algorithm])
            df_precision_results.columns = [Algorithm]
            for Algorithm_Metric in ALGORITHM_METRICS[Algorithm]:
                Precision_All_Params = [0 for _ in range(len(ALGORITHM_PARAMS[Algorithm]))]
                for param_index, Algorithm_Param in enumerate(ALGORITHM_PARAMS[Algorithm]):
                    Precision = image_retrieval_evaluation(
                        input_dataset_path=INPUT_DATASET_PATH,
                        algorithm=Algorithm,
                        algorithm_param=Algorithm_Param,
                        algorithm_metric=Algorithm_Metric,
                        number_of_retrieved_images=NUMBER_OF_RETRIEVED_IMAGES,
                        on_classes=ON_CLASSES,
                        measure=MEASURE
                    )
                    print("All " + Algorithm + Algorithm_Param + Algorithm_Metric + " computation took",
                          "--- %s seconds ---" % (time.time() - START_TIME))
                    print(Precision)
                    Precision_All_Params[param_index] = Precision
                df_precision_results[Algorithm_Metric] = Precision_All_Params
            print(df_precision_results)
            if CENTER_NORM:
                df_precision_results.to_csv(DATASET_NAME + '_' + MEASURE + '_' + Algorithm + '_center_norm.csv', header=True, index=False)
            else:
                df_precision_results.to_csv(DATASET_NAME + '_' + MEASURE + '_' + Algorithm + '_no_center_norm.csv', header=True, index=False)

    # Evaluation of the best parameters on classes
    # Result one table with P@K or mAP
    else:
        classes = [folder for folder in os.listdir(INPUT_DATASET_PATH)]
        df_precision_classes_results = pd.DataFrame(classes, columns=['Classes'])

        for Algorithm in ALGORITHMS:
            Algorithm_Metric = ALGORITHM_METRICS_BEST[Algorithm]
            Algorithm_Param = ALGORITHM_PARAMS_BEST[Algorithm]
            Precision_Classes = image_retrieval_evaluation(
                input_dataset_path=INPUT_DATASET_PATH,
                algorithm=Algorithm,
                algorithm_param=Algorithm_Param,
                algorithm_metric=Algorithm_Metric,
                number_of_retrieved_images=NUMBER_OF_RETRIEVED_IMAGES,
                test_size=TEST_SIZE,
                on_classes=ON_CLASSES,
                measure=MEASURE
            )
            print(Algorithm)
            df_method = pd.DataFrame(list(Precision_Classes.items()), columns=['Classes', Algorithm])
            df_precision_classes_results = df_precision_classes_results.merge(df_method, on=['Classes'], how='outer')
            print(df_precision_classes_results)
        if CENTER_NORM:
            df_precision_classes_results.to_csv(DATASET_NAME + '_' + MEASURE + '_classes_results_center_norm.csv', header=True, index=False)
        else:
            df_precision_classes_results.to_csv(DATASET_NAME + '_' + MEASURE + '_classes_results_no_center_norm.csv', header=True, index=False)
