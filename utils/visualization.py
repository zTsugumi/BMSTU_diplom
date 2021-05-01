import tensorflow as tf
import matplotlib.pyplot as plt


def plot_image(x_batch, y_batch, class_names, n_img):
    maxc = 3
    r = int(n_img / (maxc + 1)) + 1
    c = int(min(maxc, n_img))

    fig, axes = plt.subplots(r, c, figsize=(5, 5))
    axes = axes.flatten()
    for x, y, ax in zip(x_batch, y_batch, axes):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(x, cmap='gray')
        ax.set_title(class_names[tf.argmax(y)])
    plt.tight_layout()
    plt.show()


def plot_image_misclass(x, y_true, y_pred, class_names, n_img):
    maxc = 3
    r = int(n_img / (maxc + 1)) + 1
    c = int(min(maxc, n_img))

    idc = tf.squeeze(
        tf.where(tf.argmax(y_pred, axis=-1) != tf.argmax(y_true, axis=-1)))

    fig, axes = plt.subplots(r, c, figsize=(10, 5))
    axes = axes.flatten()
    for idx, ax in zip(idc, axes):
        ax.imshow(x[idx, ..., 0], cmap='gray')
        ax.set_axis_off()
        idx_true = tf.argmax(y_true[idx])
        class_true = class_names[idx_true]
        class_true_prob = y_pred[idx][idx_true]
        idx_pred = tf.argmax(y_pred[idx])
        class_pred = class_names[idx_pred]
        class_pred_prob = y_pred[idx][idx_pred]
        ax.set_title(
            f'Class {class_true}: {class_true_prob:.4f}\nPred {class_pred}: {class_pred_prob:.4f}')
    plt.tight_layout()
    plt.show()
