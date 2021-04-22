import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_image(x_batch, y_batch, class_names):
    fig, axes = plt.subplots(2, 3, figsize=(5, 5))
    axes = axes.flatten()
    for x, y, ax in zip(x_batch, y_batch, axes):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(x, cmap='gray')
        ax.set_title(class_names[np.argmax(y)])
    plt.tight_layout()
    plt.show()