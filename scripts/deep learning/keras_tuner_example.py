# Hyperparameter optimization using the Keras tuner
# Hyperparameters are of two types:
# 1.   Model hyperparameters which influence model selection such as the number and width of hidden layers
# 2.   Algorithm hyperparameters which influence the speed and quality of the learning algorithm such as 
#       the learning rate for Stochastic Gradient Descent (SGD) and the number of nearest neighbors 
#       for a k Nearest Neighbors (KNN) classifier

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

# Show some info
print("Tensorflow version: {:s}".format(tf.__version__))
print("Keras version: {:s}".format(keras.__version__))
print("Keras Tuner version: {:s}".format(kt.__version__))
print("GPUs:", tf.config.list_physical_devices('GPU'))


# get data
fashion_mnist = keras.datasets.fashion_mnist
(img_train, label_train), (img_test, label_test) = fashion_mnist.load_data()

# The class names are NOT included in the dataset, so 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Show some info
print("img_train shape: {} and type: {}".format(img_train.shape, img_train.dtype))      

# Normalize pixel values between 0 and 1
img_train, img_test = img_train/255.0, img_test/255.0


# Define model
# The model you set up for hypertuning is called a hypermodel. 
# You can define a hypermodel through two approaches: 
#       By using a model builder function or by subclassing the HyperModel class of the Keras Tuner API
#       By using two pre-defined HyperModel classes - HyperXception and HyperResNet for computer vision applications.

def model_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
  model.add(keras.layers.Dense(units = hp_units, activation = 'relu'))
  model.add(keras.layers.Dense(10))

  # Tune the learning rate for the optimizer 
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 

  model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                metrics = ['accuracy'])

  return model


# Hyper band tuning
# The Keras Tuner has four tuners available - RandomSearch, Hyperband, BayesianOptimization, and Sklearn
# The Hyperband tuning algorithm uses adaptive resource allocation and early-stopping to quickly converge 
#   on a high-performing model. This is done using a sports championship style bracket. 
# The algorithm trains a large number of models for a few epochs and carries forward only the top-performing half of models to the next round. 
# Hyperband determines the number of models to train in a bracket by computing 1 + logfactor(max_epochs) 
#   and rounding it up to the nearest integer. 
# To instantiate the Hyperband tuner, you must specify the hypermodel, 
#   the objective to optimize and the maximum number of epochs to train (max_epochs).
tuner = kt.Hyperband(model_builder,
                     objective = 'val_accuracy', 
                     max_epochs = 10,
                     factor = 3,
                     directory = 'tuner_example',
                     project_name = 'intro_to_kt')

# Hook up tensorboard
import os, datetime
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(logdir)                     

# Start tuning
tuner.search(img_train, label_train, epochs = 10, \
             validation_data = (img_test, label_test), \
             callbacks=[tensorboard_callback])
            #  callbacks = [ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Evaluate 
from sklearn.metrics import classification_report

tuner.search_space_summary()

best_model = tuner.get_best_models()[0]
best_model.summary()
best_model.evaluate(img_test, label_test)

predictions = best_model.predict(x=img_test, batch_size=32)
print(classification_report(label_test, predictions.argmax(axis=1), target_names=class_names))
