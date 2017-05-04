import os, pickle, sys, zipfile, io
import numpy as np
import tensorflow as tf
from scipy import misc
from glob import glob
from sklearn.metrics import roc_auc_score
import urllib.request as url_req

os.chdir("/home/bean/sushi_sandwich_local")

from utils import rescale, weight_variable, bias_variable, conv2d, max_pool, tmp_image_file, model_file, max_shape, image_url

# PARAMETERS
n_epochs = 10#int(sys.argv[1])

# CNN parameters
n_convo_layer1 = 32
n_convo_layer2 = 32
inner_layer_size = 50
percep_size = 1024

# training: batch size
batch_len = 50

### rescale the images + add rotations
def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}


if os.path.exists(tmp_image_file):
    images = pickle.load(open(tmp_image_file, "rb"))
    
else:
    print("Downloading %s..." % image_url)
    zipped_data = url_req.urlopen(image_url).read()
    zip_file = zipfile.ZipFile(io.BytesIO(zipped_data))
    zip_file.extractall()
    parent_dir = zip_file.namelist()[0].split("/")[0]
    image_path = parent_dir + "/%s/*"
    
    images = dict()
    for what in ['sushi', 'sandwich']:
        print("Loading images for '%s'..." % what)
        files = glob(image_path % what)
        full_images = [misc.imread(f, mode="L") for f in files]
        images[what] = [rescale(img, max_shape) for img in full_images]
        
        for angle in [90, 180, 270]:
            print("-- rotating by %d degrees..." % angle)
            images[what].extend([rescale(misc.imrotate(img, angle), max_shape) for img in full_images])
    
    pickle.dump(images, open(tmp_image_file, "wb"))

### model
sess = tf.Session()

# input layer
X = tf.placeholder(tf.float32, shape=[None, max_shape[0], max_shape[1], 1])
y = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)

# first convolution
W_conv1 = weight_variable([10, 10, 1, n_convo_layer1])
b_conv1 = bias_variable([n_convo_layer1])

h_conv1 = tf.nn.sigmoid(conv2d(X, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1, ksize=[1, 2, 2, 1])

# second convolution
W_conv2 = weight_variable([10, 10, n_convo_layer1, n_convo_layer2])
b_conv2 = bias_variable([n_convo_layer2])

h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2, ksize=[1, 2, 2, 1])

# first fully connected layer
W_fc1 = weight_variable([inner_layer_size * inner_layer_size * n_convo_layer2, percep_size])
b_fc1 = bias_variable([percep_size])

h_pool2_flat = tf.reshape(h_pool2, [-1, inner_layer_size * inner_layer_size * n_convo_layer2])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# second fully connected layer
W_fc2 = weight_variable([percep_size, percep_size])
b_fc2 = bias_variable([percep_size])

h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# output layer
W_fc3 = weight_variable([percep_size, 1])
b_fc3 = bias_variable([1])

logits = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-5).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

### train / test split
X_array = np.stack(images['sushi'] + images['sandwich']).reshape(-1, max_shape[0], max_shape[1], 1)
y_array = np.array([1] * len(images['sushi']) + [0] * len(images['sandwich'])).reshape(-1, 1)

test_ix = np.random.choice(len(X_array), batch_len)
train_ix = np.setdiff1d(range(len(X_array)), test_ix)

### train

for epoch in range(n_epochs):
    np.random.shuffle(train_ix)
    for batch_ix in np.array_split(train_ix, len(train_ix) / batch_len):
        sess.run(train_step, feed_dict={X: X_array[batch_ix], y: y_array[batch_ix], keep_prob: 0.9})
    
    train_sample = np.random.choice(train_ix, batch_len)
    ce, train_logits = sess.run([cross_entropy, logits], feed_dict={X: X_array[train_sample], y: y_array[train_sample], keep_prob: 1.0})
    train_auc = roc_auc_score(y_array[train_sample], train_logits)
    print("epoch {0:,}; cross-entropy: {1:.3f}".format(epoch, ce))
    
    test_logits = sess.run(logits, feed_dict={X: X_array[test_ix], y: y_array[test_ix], keep_prob: 1})
    test_auc = roc_auc_score(y_array[test_ix], test_logits)
    
    print("AUC: train {0:.2f}, test {1:.2f}".format(train_auc, test_auc))

[tf.add_to_collection('test', x) for x in [X, y, keep_prob, logits]]
saver.save(sess, model_file)