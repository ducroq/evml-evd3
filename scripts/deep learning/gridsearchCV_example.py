# Hyperparameter optimization of a Keras model using GridsearchCV
# Hyperparameters are of two types:
# 1.   Model hyperparameters which influence model selection such as the number and width of hidden layers
# 2.   Algorithm hyperparameters which influence the speed and quality of the learning algorithm such as 
#       the learning rate for Stochastic Gradient Descent (SGD) and the number of nearest neighbors 
#       for a k Nearest Neighbors (KNN) classifier
#  inspired by https://towardsdatascience.com/scikeras-tutorial-a-multi-input-multi-output-wrapper-for-capsnet-hyperparameter-tuning-with-keras-3127690f7f28

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV, cross_val_score
from scikeras.wrappers import KerasClassifier
from numpy import mean

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

# Define hypermodel
def model_builder(dense_layer_units, learning_rate):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))
  model.add(keras.layers.Dense(units = dense_layer_units, activation = 'relu'))
  model.add(keras.layers.Dense(10))
  model.compile(optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                metrics = ['accuracy'])

  return model

# Define the parameter grid that will be searched
params = {'dense_layer_units': [16, 32],
        'learning_rate': [1e-2, 1e-3]}  

# Create model, note that we need to init hyperparameters, although they are going to be varied later
model = KerasClassifier(model=model_builder, dense_layer_units=32, learning_rate=1e-2, epochs=2, batch_size=100) #, verbose=0)
print(model.get_params().keys())

# Perform gridsearch
gs = GridSearchCV(model, params, cv=2, scoring='accuracy', verbose=3)
gs_res = gs.fit(img_train, label_train)

# Let's test again, to be sure of score is correct (should be approximately equal to )
score = cross_val_score(gs_res.best_estimator_, img_train, label_train, cv=2, scoring="accuracy")
print("GridSearchCV results:")
print(f'Optimal estimator: {gs_res.best_estimator_}')
print(f'Optimal parameters: {gs_res.best_params_}')
print(f'Optimal score: {gs_res.best_score_}')
print(f"Revalidation score of best estimator: {mean(score)}")

