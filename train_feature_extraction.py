import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

# TODO: Load traffic signs data.
with open('train.p',mode='rb') as f:
    train = pickle.load(f)

X = train['features']
y = train['labels']

# TODO: Split data into training and validation sets.
n_classes = 43
(X_train,X_valid, y_train,y_valid) = train_test_split(X,y,train_size=0.7,random_state=0)

# TODO: Define placeholders and resize operation.
images = tf.placeholder(tf.float32,[None,32,32,3])
labels = tf.placeholder(tf.int64,[None])

resized_images = tf.image.resize_images(images,tf.constant([227,227],name='image_resize'))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized_images, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1],n_classes)
fc8W  = tf.Variable(tf.truncated_normal(shape,0,0.1, dtype=tf.float32))
fc8b  = tf.Variable(tf.zeros([n_classes], dtype=tf.float32))

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
logits = tf.nn.bias_add(tf.matmul(fc7,fc8W),fc8b)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels)
loss = tf.reduce_mean(cross_entropy)


optimizer = tf.train.AdamOptimizer()
optimization = optimizer.minimize(loss,var_list=[fc8W,fc8b])

correct_predictions = tf.equal(tf.arg_max(logits,1),labels)
accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))

# Initilization operation should be the last added node to the graph.
# Do not add any nodes after this line.
initialization = tf.global_variables_initializer()

# TODO: Train and evaluate the feature extraction model.
batch_sz = 64
def evaluate(sessn,X,y):
    n_examples = X.shape[0]
    
    net_accuracy = 0
    
    for offset in range(0,n_examples,batch_sz):
        X_batch = X[offset:offset+batch_sz]
        y_batch = y[offset:offset+batch_sz]
        
        batch_accuracy = sessn.run(accuracy,feed_dict={images:X_batch,labels:y_batch})
        net_accuracy += batch_accuracy * batch_sz
        
    return net_accuracy / n_examples

# Run training.
n_epoch = 10
with tf.Session() as s:
    s.run(initialization)
    
    for e in range(n_epoch):
        X_train,y_train = shuffle(X_train, y_train)
        
        n_examples = X_train.shape[0]
        
        for offset in range(0,n_examples,batch_sz):
            X_batch = X_train[offset:offset+batch_sz]
            y_batch = y_train[offset:offset+batch_sz]
            
            s.run(optimization,feed_dict={images:X_batch, labels:y_batch})
            
        train_accuracy = evaluate(s,X_train,y_train)
        validation_accuracy = evaluate(s,X_valid,y_valid)
        
        print("Epochs {}, Training accuracy {:.3f}, Validation accuracy {:.3f}".format(e+1,train_accuracy,validation_accuracy))
    
    
    
