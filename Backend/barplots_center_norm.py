import numpy as np
import matplotlib.pyplot as plt

N=5
DATASET_NAME = 'wang'

if DATASET_NAME == 'wang':
    a = [0.740068, 0.812039, 0.815864, 0.906293, 0.909234]
    b = [0.746387, 0.811845, 0.846155, 0.915427, 0.918274]
    labels = ['AlexNet', 'VGG', 'MobileNet', 'ResNet', 'EfficientNet']

    ind = np.arange(N)
    width=0.35

    fig, ax = plt.subplots()
    b1 = ax.bar(ind, a, width, color='gold')
    b2 = ax.bar(ind+width, b, width, color='dodgerblue')
    color = {'cos': 'gold', 'cos center+norm': 'dodgerblue'}

    legend_labels = ['cos', 'cos center+norm']
    legend_handles = [plt.Rectangle((0,0),1,1, color=color[legend_label]) for legend_label in legend_labels]
    ax.legend(legend_handles, legend_labels)

    ax.set_xticks(ticks=ind+width/2, labels=labels)
    ax.set_title('Comparison of the preprocessing on the Wang dataset')
    plt.ylabel('mAP')
    plt.show()

elif DATASET_NAME == 'patterns':
    a = [0.594306, 0.706779, 0.665886, 0.735631, 0.718587]
    b = [0.582159, 0.718422, 0.701564, 0.748859, 0.728154]

    labels = ['AlexNet', 'VGG', 'MobileNet', 'ResNet', 'EfficientNet']

    ind = np.arange(N)
    width=0.35

    fig, ax = plt.subplots()
    b1 = ax.bar(ind, a, width, color='gold')
    b2 = ax.bar(ind+width, b, width, color='dodgerblue')
    color = {'cos': 'gold', 'cos center+norm': 'dodgerblue'}

    legend_labels = ['cos', 'cos center+norm']
    legend_handles = [plt.Rectangle((0,0),1,1, color=color[legend_label]) for legend_label in legend_labels]
    ax.legend(legend_handles, legend_labels, framealpha=0.5)

    ax.set_xticks(ticks=ind+width/2, labels=labels)
    ax.set_title('Comparison of the preprocessing on the Patterns dataset')
    plt.ylabel('mAP')
    plt.show()