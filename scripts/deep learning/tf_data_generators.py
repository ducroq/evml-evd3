import logging
import os, glob
from tensorflow import keras
from tensorflow import get_logger

get_logger().setLevel(logging.ERROR)
data_path = r'C:\Users\scbry\OneDrive - HAN\data\EVML\gesture_data'
batch_size = 32
image_shape = (224, 224, 3)

# Show some info
print("\n")
print("[INFO] Keras version: {:s}".format(keras.__version__))

# Set-up data
file_list = [f for f in glob.iglob(os.path.sep.join([data_path, '**', '*.png']), recursive=True) \
        if (os.path.isfile(f) and "annotated" not in f)]
print("[INFO] {} images found".format(len(file_list)))

label_names = [os.path.split(f)[-1] for f in glob.iglob(os.path.sep.join([data_path, '*'])) \
        if os.path.isdir(f)]
print("[INFO] {} labels found: {}".format(len(label_names), label_names))
print("\n")

# Using image data generator and directory iterator
# See: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image
# Note that tf.keras.preprocessing.image is depricated
data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, \
        preprocessing_function=None, validation_split=0.2) #, rotation_range=20)

images, labels = next(data_gen.flow_from_directory(data_path))
print("[INFO] Image set shape: {} and type: {}:".format(images.shape, images.dtype))

print("[INFO] Setting up training data")
train_ds = keras.preprocessing.image.DirectoryIterator(data_path, data_gen, \
        target_size=image_shape[0:2], batch_size=batch_size, subset="training")
# images, labels = next(train_ds)

print("[INFO] Setting up validation data")
val_ds = keras.preprocessing.image.DirectoryIterator(data_path, data_gen, \
        target_size=image_shape[0:2], batch_size=batch_size, subset="validation")

print("\n\n\n")

# Using image data generator and flow_from_directory
datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=.2)
dataflow_kwargs = dict(target_size=image_shape, batch_size=batch_size,
                   interpolation="bilinear")

print("[INFO] Setting up training generator")
train_generator = datagen.flow_from_directory(
    data_path, subset="training", shuffle=False, **dataflow_kwargs)
    
print("[INFO] Setting up validation generator")
valid_generator = datagen.flow_from_directory(
    data_path, subset="validation", shuffle=False, **dataflow_kwargs)

print("\n\n\n")

# Using image_dataset_from_directory is the preferred method, since tf.keras.preprocessing.image is depricated
train_ds = keras.preprocessing.image_dataset_from_directory(data_path,
        validation_split=0.2, subset="training", seed=123, image_size=image_shape[0:2], batch_size=batch_size)
val_ds = keras.preprocessing.image_dataset_from_directory(data_path,
        validation_split=0.2, subset="validation", seed=123, image_size=image_shape[0:2], batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)

