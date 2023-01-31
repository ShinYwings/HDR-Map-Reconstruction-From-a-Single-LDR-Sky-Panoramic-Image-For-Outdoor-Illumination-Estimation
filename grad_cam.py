import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show(filters, mpath, index, nx=8, show = True, save = False):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN = filters.shape[0]
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i], interpolation='nearest')
        # ax.set_title(f"elevation:{sunpose[i]}", fontsize=5)
    
    if show:
        plt.show()

    if save:
        plt.savefig("{}/{}.png".format(mpath, index))
    
    plt.clf()

def layer(y_c, A_k):

    grad = tf.gradients(y_c, A_k)[0]

    # Global average pooling
    weights = tf.reduce_mean(grad, axis = (1, 2))
    cam = tf.einsum('bc,bwhc->bwh', weights, A_k)
    
    # 계산된 weighted combination 에 ReLU 적용
    cam = tf.nn.relu(cam)

    # get better result without normalization
    # cam = tf.divide(cam, (tf.reduce_max(cam) + 1e-10))
    
    cam = tf.expand_dims(cam, axis=-1)

    return cam