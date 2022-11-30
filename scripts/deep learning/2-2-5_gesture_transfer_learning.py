import os, glob
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import kerastuner as kt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from numpy import round
from seaborn import heatmap

data_path = r'C:\Users\scbry\OneDrive - HAN\data\EVML\gesture_data'
project_name = 'gesture_transfer_learning'
do_data_augmentation = True
image_shape = (224, 224, 3)
batch_size = 32
nr_of_epochs = 10

# Show some general info
print("[INFO] Tensorflow version: {:s}".format(tf.__version__))
print("[INFO] Keras version: {:s}".format(keras.__version__))
print("[INFO] Keras Tuner version: {:s}".format(kt.__version__))

physical_devices = tf.config.list_physical_devices()
print("[INFO] Physical devices:")
for p in physical_devices:
    print("\t{}".format(p.device_type))

# Hook up Tensorboard
root_logdir = os.path.join(os.getcwd(), "my_logs", project_name + '_logs')
run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
run_logdir = os.path.join(root_logdir, run_id)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# Set-up data
file_list = [f for f in glob.iglob(os.path.sep.join([data_path, '**' , '*.png' ]), recursive=True) 
        if (os.path.isfile(f) and not "annotated" in f)] 
print("[INFO] {} images found".format(len(file_list)))

label_names = [os.path.split(f)[-1] for f in glob.iglob(os.path.sep.join([data_path, '*'])) 
        if os.path.isdir(f)]
print("[INFO] {} labels found: {}".format(len(label_names), label_names))

# See: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image
#  and https://www.tensorflow.org/hub/tutorials/tf2_image_retraining
datagen_kwargs = dict(rescale=1./255, preprocessing_function=None, validation_split=0.2)
data_generator = keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
# Note that there should be preprocessing, according to
#  https://keras.io/api/applications/mobilenet/#mobilenetv2-function
# However, that doesn't do much good in practice. why?

dataflow_kwargs = dict(directory=data_path, target_size=image_shape[:-1], batch_size=batch_size, 
    color_mode='rgb', interpolation="bilinear")

print("[INFO] Setting up training generator")
train_generator = data_generator.flow_from_directory(
    subset="training", shuffle=True, **dataflow_kwargs)
if do_data_augmentation:
    train_data_generator = keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs,
        rotation_range=90, horizontal_flip=True,
        width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.1, zoom_range=0.2)

print("[INFO] Setting up validation generator")
valid_generator = data_generator.flow_from_directory(
    subset="validation", shuffle=False, **dataflow_kwargs)
# No shuffling, otherwise confusion matrix computation gets shuffled input too

# Define the model
def model_builder(hp):
    # Import MobileNetV2 model and discards the last 1000 neuron layer.
    base_model = keras.applications.MobileNetV2(
        input_shape=image_shape, weights='imagenet',include_top=False)
    base_model.trainable = False
    # Playing around with freezing/unfreezing could be interesting

    # Set-up the model
    model = keras.models.Sequential()
    model.add(base_model)
    model.add(keras.layers.MaxPool2D(2,2))
    # model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(len(label_names), activation="softmax"))

    # Compile the model
    # Tune the learning rate for the optimizer 
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])     
    optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate, clipvalue=1.0)
    # optimizer = keras.optimizers.SGD(lr=hp_learning_rate,momentum=0.99)

    model.compile(optimizer=optimizer, 
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'])

    return model

# Training and tuning
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size
tuner = kt.Hyperband(model_builder,
    objective = 'val_accuracy',
    directory = 'tuner',
    project_name = project_name,
    max_epochs = 10, factor = 3)
tuner.search(train_generator,
    epochs=nr_of_epochs, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator, validation_steps=validation_steps,
    callbacks=[tensorboard_cb])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
# print([hp for hp in best_hps])
# best_hps.Choice = {'learning_rate', values=1e-3}
tuner.search_space_summary()
# Alternatively, training without tuning
model = model_builder(best_hps)
history = model.fit(
    train_generator,
    epochs=nr_of_epochs, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator, validation_steps=validation_steps,
    callbacks=[tensorboard_cb]).history

# Evaluate model
model = tuner.get_best_models()[0]
model.summary()

predictions = model.predict(valid_generator)
most_probable_predictions = predictions.argmax(axis=1)
print(classification_report(valid_generator.labels, most_probable_predictions, target_names=label_names))

# Compute confusion matrix
cm = round(confusion_matrix(valid_generator.labels, most_probable_predictions, normalize='true'),1)

# Plot results
plt.figure()
heatmap(cm, cmap=plt.cm.Blues, annot=True, xticklabels=label_names, yticklabels=label_names)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1)
ax.set_ylabel("Loss")
ax.set_xlabel("Training Steps")
# plt.ylim([0,2])
ax.plot(history["loss"])
ax.plot(history["val_loss"])
ax.legend(['train', 'val'], loc='lower left')

fig1, ax1 = plt.subplots(1, 1)
ax1.set_title('Learning Curves')
ax1.set_xlabel('training set size')
ax1.set_ylabel('Accuracy')
ax1.plot(history["accuracy"])
ax1.plot(history["val_accuracy"])
ax1.legend(['train', 'val'], loc='lower right')

plt.show()