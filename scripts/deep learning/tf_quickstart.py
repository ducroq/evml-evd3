import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load fashion MNIST, which is a drop-in replacement of MNIST
# Note that on first time, downloading takes very long and fails sometimes
# (70,000 grayscale images of 28 × 28 pixels each, with 10 classes)
fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Alternatively, download the following files
# https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
# https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
# https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
# https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
# import os
# import gzip
# dirname = os.path.join('datasets', 'fashion-mnist')
# files = [
#   'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
#   't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
# ]
# paths = [os.path.join(dirname, file) for file in files]

# with gzip.open(paths[0], 'rb') as lbpath:
#     y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

# with gzip.open(paths[1], 'rb') as imgpath:
#     x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

# with gzip.open(paths[2], 'rb') as lbpath:
#     y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

# with gzip.open(paths[3], 'rb') as imgpath:
#     x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

# (X_train_full, y_train_full), (X_test, y_test) = \
#                (np.copy(x_train).astype(float), np.copy(y_train)), \
#                (np.copy(x_test).astype(float), np.copy(y_test))

# The class names are NOT included in the dataset, so 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Show some info
print("Tensorflow version: {:s}".format(tf.__version__))
print("Keras version: {:s}".format(keras.__version__))
print("X_train_full shape: {} and type: {}".format(X_train.shape, X_train.dtype))

# Scale pixel values down to the 0–1 range
# It's important that the training set and the testing set be preprocessed
X_train, X_test = X_train/255.0, X_test/255.0

# Display the first 25 images from the training set and display the class name below each image
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show(block=True)

# Create a model
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(64, activation='relu'),
#     ##  tf.keras.layers.Dropout(0.2),
#     keras.layers.Dense(10)
# ])

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# Show some info
model.summary()
# Note that Dense layers often have a lot of parameters. For example, the first hidden
# layer has 28×28× density = 784 × 128 connection weights, plus 128 bias terms
# This gives the model quite a lot of flexibility to fit the training
# data, but it also means that the model runs the risk of overfitting, especially when you
# do not have a lot of training data.

# Compile the model
model.compile(optimizer='adam', # 'sgd'
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Feed the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, shuffle=True)

# Show history
print(history.history.keys())

# Plot learning curves
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='loss (training data)')
ax.plot(history.history['val_loss'], label='loss (validation data)')
ax.plot(history.history['accuracy'], label='accuracy (training data)')
ax.plot(history.history['val_accuracy'], label='accuracy (validation data)')
ax.set_title('Learning curves')
ax.set_ylabel('value')
ax.set_xlabel('No. epoch')
ax.grid(True)
ax.set_ylim(0,1)
ax.legend(loc="lower right")
plt.show()

# Check performance on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Compute confusion matrix
from sklearn.metrics import confusion_matrix

predictions = model.predict(X_test)
most_probable_predictions = np.argmax(predictions, axis=1)
cm = np.round(confusion_matrix(y_test, most_probable_predictions, normalize='true'),1)

# Plot confusion matrix
import seaborn as sns

plt.figure()
ax4 = sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, xticklabels=class_names, yticklabels=class_names)
plt.tight_layout()
plt.show()
