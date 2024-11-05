import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Create a simple 6x6 input image
input_image = np.array([
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 0]
], dtype=np.float32)

# Or load an image as a grayscale numpy array
input_image = cv2.imread(r'C:\Users\scbry\OneDrive - HAN\Onderwijs\evml-evd3\scripts\deep learning\mus.jpg', cv2.IMREAD_GRAYSCALE)
input_image = input_image / 255.0  # TensorFlow expects normalized images

# Define a 3x3 Laplacian operator for edge detection
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]], dtype=np.float32)
# # Define a 3x3 Scharr operator for horizontal edge detection
# kernel = np.array([
#     [ 3,  10,  3],
#     [ 0,   0,  0],
#     [-3, -10, -3]
# ], dtype=np.float32)

# Reshape input for TensorFlow
kernel_size = kernel.shape[0]
height, width = input_image.shape
input_tensor = tf.reshape(input_image, [1, height, width, 1])
kernel_tensor = tf.reshape(kernel, list(kernel.shape) + [1, 1])

# Create convolutional layer
stride = 2
conv_layer = tf.keras.layers.Conv2D(
    filters=1,
    kernel_size=kernel_size,
    strides=(stride, stride),  # Set stride for height and width
    padding='valid',  # 'valid' (no padding) or 'same' (pad to keep same size)
    use_bias=False
)

# Set the kernel weights manually
conv_layer.build((None, height, width, 1))
conv_layer.set_weights([kernel_tensor.numpy()])

# Apply convolution
feature_map = conv_layer(input_tensor)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.imshow(input_image, cmap='gray')
ax1.set_title('Input Image')

ax2.imshow(kernel, cmap='gray')
ax2.set_title('Kernel (Edge Detection)')

ax3.imshow(feature_map[0, :, :, 0], cmap='gray')
ax3.set_title('Feature Map')

plt.tight_layout()
plt.show()

# # Print feature map values
# print("\nFeature Map Values:")
# print(feature_map[0, :, :, 0].numpy())