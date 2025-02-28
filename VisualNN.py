import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, UpSampling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import numpy as np

from scipy.stats import mode

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold

'''
IMPORT DATA:
'''
# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Original Y Training | Labels:", np.unique(y_train))
print("Original Y Test | Labels:", np.unique(y_test))


# Flatten and normalize images
#x_data = np.concatenate((x_train, x_test))  # Merge all data
#y_data = np.concatenate((y_train, y_test))
#print("x_data Original shape:", x_data)

'''
ADD LABELS TO DATA
'''
# Split the data by labels where train is digits 0-4 and test is digits 5-9
#train_mask = np.isin(y_data, [0, 1, 2, 3, 4])
#test_mask = np.isin(y_data, [5, 6, 7, 8, 9])
train_mask = np.isin(y_train, [0, 1, 2, 3, 4])
test_mask = np.isin(y_test, [5, 6, 7, 8, 9])


x_train = x_train[train_mask]
y_train = y_train[train_mask]
x_test = x_test[test_mask]
y_test = y_test[test_mask]

print("Filtered Y Training | Labels:", np.unique(y_train))
print("Filtered Y Test | Labels:", np.unique(y_test))

# Remap the labels 5-9 to 0-4 for the test set
label_mapping = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4}
y_test = np.array([label_mapping[label] for label in y_test])

print("Remapped Y Test | Labels:", np.unique(y_test))

'''
PREPROCESS DATA:
'''
# Normalize the images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print("x_train Normalize shape:", x_train.shape)
print("x_test Normalize shape:", x_test.shape)

# Expand dimensions to match the input shape of the pre-trained model
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
print("x_train Dim shape:", x_train.shape)
print("x_test Dim shape:", x_test.shape)

# Resize images to (32, 32, 3)
x_train = tf.image.resize(x_train, (32, 32))
x_test = tf.image.resize(x_test, (32, 32))
print("x_train re-size shape:", x_train.shape)
print("x_test re-size shape:", x_test.shape)

# Convert to 3 channels
x_train = tf.image.grayscale_to_rgb(x_train)
x_test = tf.image.grayscale_to_rgb(x_test)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 5)  # Use 5 classes since we are focusing on digits 0-4
y_test = to_categorical(y_test, 5)  # Use 5 classes since we are focusing on digits 0-4

print("Preprocess Y Train Labels:", np.unique(y_train))
print("Preprocess Y Test Labels:", np.unique(y_test))
print("\n")

'''
TRANSFER MODEL:
- VGG16
- ResNet50
- EfficientNetB0: is a lightweight and efficient model that often provides good 
                  performance with fewer parameters compared to other models.
'''

# Define the model architecture
def create_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(5, activation='softmax')(x)  # Change to 5 classes
    model = Model(inputs=inputs, outputs=outputs)
    return model

# K-Fold Cross-Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_accuracy = 0.0
best_model_path = 'C:/ML2/ML2_HOMEWORK/mnist_EfficientNetB0.h5'

# Iterate over each fold
for fold, (train_index, val_index) in enumerate(kf.split(x_train)):
    print(f"Training Fold {fold + 1}...")
    x_train_fold = tf.gather(x_train, train_index)
    x_val_fold = tf.gather(x_train, val_index)
    y_train_fold = tf.gather(y_train, train_index)
    y_val_fold = tf.gather(y_train, val_index)

    model = create_model(input_shape=(32, 32, 3))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train_fold, y_train_fold, epochs=10, batch_size=32)

    val_loss, val_accuracy = model.evaluate(x_val_fold, y_val_fold)
    print(f"Fold {fold + 1} | Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_fold = fold + 1
        y_pred = model.predict(x_val_fold)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val_fold, axis=1)
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        print(f"Confusion Matrix for Fold {fold + 1}:\n{cm}")
        model.save(best_model_path)
        print(f"Saved new best model with accuracy: {val_accuracy:.4f}")

print(f"The best fold was Fold {best_fold} with accuracy: {best_accuracy:.4f}")