import numpy as np
import matplotlib.pyplot as plt

from descriptors.datasets import MPEG7
from descriptors.rectangularity.rectangularity import rectangularity

plt.style.use('seaborn')

methods = ['Minimum Bounding Rectangle (MBR)',
           'Rectangular Discrepancy',
           'Robust MBR',
           'Agreement method',
           'Moments method']

kinds = ['apple', 'bottle', 'cellular_phone', 'Bone', 'camel']
kind = 'bottle'

images = MPEG7.load_data(kind=kind)

x = np.arange(1, len(images) + 1)
methods_descriptors = {}

for method in methods:

    descriptors = np.zeros(len(images))

    for i, img in enumerate(images.values()):
        descriptors[i] = rectangularity(img, method)

    plt.scatter(x, descriptors, color='black')
    plt.xticks(x)

    round_descriptors = np.around(descriptors, decimals=3)
    for i, p in enumerate(round_descriptors):
        plt.annotate(p, (x[i], descriptors[i]), color='blue')

    plt.title(f"{method} for kind: {kind}")
    plt.ylim(-0.1, 1.1)
    plt.show()

    # print(method, descriptors)

    methods_descriptors[method] = descriptors

for method in methods_descriptors.keys():
    plt.scatter(x, methods_descriptors[method], label=method)

plt.xticks(x)
plt.title(f'All methods for kind: {kind}')
plt.ylim(-0.1, 1.1)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.show()
