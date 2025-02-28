import tensorflow as tf

# Load the TensorBoard notebook extension (remove this line if not using Jupyter Notebook)

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
import datetime
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import confusion_matrix

# Clear any logs from previous runs
import shutil
import os

if os.path.exists('./logs/'):
    shutil.rmtree('./logs/')

# Enable eager execution
tf.config.run_functions_eagerly(True)

'''
IMPORT DATA:
'''
# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Original Y Training | Labels:", np.unique(y_train))
print("Original Y Test | Labels:", np.unique(y_test))

# Flatten and normalize images
x_data = np.concatenate((x_train, x_test))  # Merge all data
y_data = np.concatenate((y_train, y_test))

'''
ADD LABELS TO DATA
'''
# Split the data by labels where train is digits 0-4 and test is digits 5-9
#train_mask = np.isin(y_data, [0, 1, 2, 3, 4])
test_mask = np.isin(y_data, [5, 6, 7, 8, 9])

#x_train = x_data[train_mask]
#y_train = y_data[train_mask]
x_test = x_data[test_mask]
y_test = y_data[test_mask]

#print("Filtered Y Training | Labels:", np.unique(y_train))
print("Filtered Y Test | Labels:", np.unique(y_test))

# Remap the labels 5-9 to 0-4 for the test set
label_mapping = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4}
y_test = np.array([label_mapping[label] for label in y_test])

print("Remapped Y Test | Labels:", np.unique(y_test, return_counts=True))

'''
PREPROCESS DATA:
'''
# Normalize the images
x_test = x_test.astype('float32') / 255.0
print("x_test Normalize shape:", x_test.shape)

# Expand dimensions to match the input shape of the pre-trained model
x_test = np.expand_dims(x_test, axis=-1)
print("x_test Dim shape:", x_test.shape)

# Resize images to (32, 32, 3)
x_test = tf.image.resize(x_test, (32, 32))
print("x_test re-size shape:", x_test.shape)

# Convert to 3 channels
x_test = tf.image.grayscale_to_rgb(x_test)
print("x_test shape:", x_test.shape)

print("Current y_test:", y_test)
# Convert labels to one-hot encoding
y_test = to_categorical(y_test, 5)  # Use 5 classes since we remapped to 0-4

print("Preprocess Y Test Labels:", np.unique(y_test))
print("\n")



'''
DATA AUGMENTATION:
- gat me around a 11% increase in accuracy now at around 41%
'''
# Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Fit the data generator to your training data
datagen.fit(x_test)

'''
REUSE MODEL WITH TEST DATASET THIS TIME
'''

# Load the pre-trained model
base_model = tf.keras.models.load_model('C:/ML2/ML2_HOMEWORK/mnist_EfficientNetB0.h5')
base_model.summary()

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add new layers on top of the base model
inputs = base_model.input
x = base_model.output
x = Flatten()(inputs)
x = Dense(128, activation='relu')(x)
outputs = Dense(5, activation='softmax')(x)   # Change to 5 classes
model = Model(inputs=inputs, outputs=outputs)

# Compile the model with a new optimizer
new_optimizer = Adam(learning_rate=1e-5)
model.compile(optimizer=new_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Tensorboard Callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', 
    patience=1, # parameter is set to 5, which means that the training will stop if the loss does not improve for 5 consecutive epochs.
    restore_best_weights=False # parameter is set to True, ensuring that the model weights are restored to the best state observed during training.
)

# Fine-tune the model on the augmented training dataset
history = model.fit(datagen.flow(x_test, y_test, batch_size=32), 
                    epochs=50, 
                    #validation_data=.2, 
                    callbacks=[early_stopping, tensorboard_callback])

# Evaluate the fine-tuned model on the test dataset
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Predict the classes for the test dataset
y_test_pred = model.predict(x_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true_classes = np.argmax(y_test, axis=1)

# To run Tensorboard, use the following command in your terminal:
# tensorboard --logdir logs/fit

# Generate the confusion matrix
test_cm = confusion_matrix(y_test_true_classes, y_test_pred_classes)
print(f"Confusion Matrix for Test Data:\n{test_cm}")

# Check the distribution of predicted classes
print("Predicted class distribution:", np.unique(y_test_pred_classes))

# Save the model
model.save('C:/ML2/ML2_HOMEWORK/mnist_EfficientNetB0_UPDATED.h5')