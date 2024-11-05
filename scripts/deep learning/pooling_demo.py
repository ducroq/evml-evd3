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
input_image = input_image / 255.0  # normalize to [0,1] range

# Reshape input for TensorFlow
height, width = input_image.shape
input_tensor = tf.reshape(input_image, [1, height, width, 1])

# Create pooling layers
pool_size = (2, 2)      # Size of the pooling window
stride = 2              # How many pixels to skip

max_pool_layer = tf.keras.layers.MaxPooling2D(
    pool_size=pool_size,
    strides=stride
)

avg_pool_layer = tf.keras.layers.AveragePooling2D(
    pool_size=pool_size,
    strides=stride
)

# Apply pooling
max_pool_output = max_pool_layer(input_tensor)
avg_pool_output = avg_pool_layer(input_tensor)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.imshow(input_image, cmap='gray')
ax1.set_title(f'Input Image\n{input_image.shape}')

ax2.imshow(max_pool_output[0, :, :, 0], cmap='gray')
ax2.set_title(f'Max Pooling\n{max_pool_output.shape[1:3]}')

ax3.imshow(avg_pool_output[0, :, :, 0], cmap='gray')
ax3.set_title(f'Average Pooling\n{avg_pool_output.shape[1:3]}')

plt.tight_layout()
plt.show()

# Print some pooling values for a small region
print("\nExample pooling operation on a 2x2 region:")
y, x = 0, 0  # top-left corner
region = input_image[y:y+2, x:x+2]
print(f"Input region:\n{region}")
print(f"Max value: {np.max(region):.3f}")
print(f"Average value: {np.mean(region):.3f}")