import time
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread
from alexnet import AlexNet

sign_names = pd.read_csv('signnames.csv')
nb_classes = 43

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int8,[None])
resized = tf.image.resize_images(x, (227, 227))

# NOTE: By setting `feature_extract` to `True` we return
# the second to last layer.
fc7 = AlexNet(resized, feature_extract=True)
# TODO: Define a new fully connected layer followed by a softmax activation to classify
# the traffic signs. Assign the result of the softmax activation to `probs` below.
# HINT: Look at the final layer definition in alexnet.py to get an idea of what this
# should look like.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

# Define custom fuly connected layer for our application.
fc8W  = tf.Variable(tf.truncated_normal(list(shape),0,0.1))
fc8b  = tf.Variable(tf.zeros([nb_classes]))

# Define softmax output layer.
logits = tf.nn.bias_add(tf.matmul(fc7,fc8W),fc8b)
probs = tf.nn.softmax(logits)

# Define loss funtion
one_hot_y = tf.one_hot(y,n_classes,on_value=1,off_value=0)
cross_entropy = sparse_softmax_cross_entropy_with_logits(logits,one_hot_y)
training_loss = tf.reduce_mean(cross_entropy)

# Optimizer
optimizer = tf.train.AdamOptimizer()
optimization = optimizer.minimize(cross_entropy,var_list=[fc8W,fc8b])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))
