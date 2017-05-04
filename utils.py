import tensorflow as tf
import numpy as np
from scipy import misc

# PARAMETERS
image_url = "http://research.us-east-1.s3.amazonaws.com/public/sushi_or_sandwich_photos.zip"
tmp_image_file = "images.pkl"
model_file = "model"

# rescale images to this size
max_shape = np.array((200, 200))

# FUNCTIONS
def rescale(image, max_shape):
    
    max_shape = np.array(max_shape)
    img_shape = np.array(image.shape)[:2]
    
    # rescale the image
    ratios = np.array(img_shape) / max_shape
    largest_dim = np.argmax(ratios)
    factor = max_shape[largest_dim] / image.shape[largest_dim]
    target_shape = np.round((factor * img_shape)).astype(int)
    rescaled = misc.imresize(image, target_shape)
    
    # pad it
    pads = [(0,0)] * image.ndim
    #pads = max_shape - np.array(rescaled.shape[:2])
    pads[1 - largest_dim] = tuple(max_shape - np.array(rescaled.shape[:2]))
    
    return np.pad(rescaled, pads, 'constant')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return(tf.Variable(initial))


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return(tf.Variable(initial))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def max_pool(x, ksize=[1, 2, 2, 1]):
    return tf.nn.max_pool(x, ksize=ksize, strides=[1, 1, 1, 1], padding='SAME')
