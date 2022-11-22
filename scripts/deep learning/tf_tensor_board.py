import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Start tensorboard server from the terminal
# % tensorboard --logdir "deep learning/my_logs"

# Show some info
print("Tensorflow version: {:s}".format(tf.__version__))
print("Keras version: {:s}".format(keras.__version__))
physical_devices = tf.config.list_physical_devices()
print(physical_devices)

root_logdir = os.path.join(os.curdir, "my_logs")
run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
run_logdir = os.path.join(root_logdir, run_id)
print("Tensor board logging folder: {}".format(run_logdir))

# Load fashion MNIST, which is a drop-in replacement of MNIST
# Note that on first time, downloading takes very long and fails sometimes
# (70,000 grayscale images of 28 × 28 pixels each, with 10 classes)
keras.datasets.fashion_mnist.load_data()
fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# The class names are NOT included in the dataset, so 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Show some info
print("X_train_full shape: {} and type: {}".format(X_train.shape, X_train.dtype))

# Scale pixel values down to the 0–1 range
# It's important that the training set and the testing set be preprocessed
X_train, X_test = X_train/255.0, X_test/255.0

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(300, activation="elu"))
# model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(100, activation="elu"))
# model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(10, activation="softmax"))

# Show some info
model.summary()
# Note that Dense layers often have a lot of parameters. For example, the first hidden
# layer has 28×28× density = 784 × 128 connection weights, plus 128 bias terms
# This gives the model quite a lot of flexibility to fit the training
# data, but it also means that the model runs the risk of overfitting, especially when you
# do not have a lot of training data.

# Compile the model
optimizer = keras.optimizers.Adam(clipvalue=1.0)
model.compile(optimizer=optimizer, # 'sgd'
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Feed the model
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, shuffle=True, callbacks=[tensorboard_cb])

# Run tensorboard from the command line 
# tensorboard --logdir my_logs

# Check performance on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Compute confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

predictions = model.predict(X_test)
most_probable_predictions = predictions.argmax(axis=1)
print(classification_report(y_test, most_probable_predictions, target_names=class_names))


# predictions = model.predict(X_test)
# most_probable_predictions = np.argmax(predictions, axis=1)
# 
# Plot confusion matrix
from seaborn import heatmap

cm = np.round(confusion_matrix(y_test, most_probable_predictions, normalize='true'),1)
plt.figure()
ax4 = heatmap(cm, cmap=plt.cm.Blues, annot=True, xticklabels=class_names, yticklabels=class_names)
plt.tight_layout()
plt.show()
