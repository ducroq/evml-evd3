import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Show some info
print("Tensorflow version: {:s}".format(tf.__version__))
print("Keras version: {:s}".format(keras.__version__))

# Load the model
# Note that on first time, downloading takes very long and fails sometimes
model = keras.applications.VGG16()

# summarize filter shapes
for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)

# Example of visualizing some filters
# retrieve weights from the second hidden layer
filters, biases = model.layers[1].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
plt.figure()
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = plt.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		ax.imshow(f[:, :, j], cmap='gray')
		ix += 1

# Example of visualizing feature maps
# redefine model to output right after the first hidden layer
import os
ixs = [2, 5, 9, 13, 17]
outputs = [model.layers[i].output for i in ixs]
model = keras.models.Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape
img = keras.preprocessing.image.load_img(r'scripts\deep learning\mus.jpg', target_size=(224, 224))
# convert the image to an array
img = keras.preprocessing.image.img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = np.expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = keras.applications.vgg16.preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
plt.figure()
# plot the output from each block
square = 8
for fmap in feature_maps:
	# plot all 64 maps in an 8x8 squares
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax1 = plt.subplot(square, square, ix)
			ax1.set_xticks([])
			ax1.set_yticks([])
			# plot filter channel in grayscale
			ax1.imshow(fmap[0, :, :, ix-1], cmap='gray')
			ix += 1

plt.show()
