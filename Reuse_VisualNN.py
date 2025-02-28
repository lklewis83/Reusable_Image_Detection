
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import fitz  # PyMuPDFimport fitz  # PyMuPDF
import cv2
import os
import numpy as np
import random
from sklearn.utils.class_weight import compute_class_weight

# Define a mapping for the characters based on your knowledge of the order in images
char_labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
char_order = ['A', 'B', 'C', 'D', 'E']  

# Directory for images
directory = 'C:/ML2/ML2_HOMEWORK/imageData'

def extract_images(pdf_path):
    doc = fitz.open(pdf_path)
    image_list = []
    output_dir = directory

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for page_number in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(page_number)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"image_page_{page_number}_{img_index}.{image_ext}"
            with open(os.path.join(output_dir, image_filename), 'wb') as img_file:
                img_file.write(image_bytes)
            image_list.append(image_filename)
    return image_list

# Call the function with the path to your PDF
extracted_images = extract_images('C:/ML2/ML2_HOMEWORK/CapitalLetters.pdf')
print(extracted_images)

# REMOVE EXTRA FILES
for filename in os.listdir(directory):
    print("Filename:",filename)
    if filename.endswith(".jpeg"):  # Adjust the file extension if necessary
        # Extract the last character before '.jpeg'
        last_char = filename[-6]  # The last character before '.jpeg'

        print(f"Filename: {filename} - Last Character: {last_char}")  # Debug statement

        # Check if the last character before '.jpeg' is '0'
        if last_char == '0':
            file_path = os.path.join(directory, filename)
            print(f"Deleting: {file_path}")  # Debug statement
            os.remove(file_path)  # Uncomment this line to actually delete the files
        else:
            print(f"Keeping: {filename}")  # Debug statement

'''
The shape of your dataset showing 784 columns indicates that each image has been flattened 
into a one-dimensional array of 784 elements. This occurs because the images have been resized 
to a 28x28 pixel format, and 28 * 28 = 784. Flattening is a common preprocessing step for image 
data when preparing it for input into machine learning models, especially neural networks, which 
often require input data to be in a vector form.

Here’s a breakdown of what's happening:

- Image Resizing: Each character image is resized to 28x28 pixels.

- Flattening: The 28x28 pixel image is then flattened into a one-dimensional array with 784 elements. 
This means each pixel's grayscale value becomes a separate feature in the dataset.

- Storing in X: These flattened arrays are appended to list X, which is then converted to a numpy array. 
Thus, the shape of X becomes (n, 784), where n is the number of images, and 784 represents the pixel values of each image.
'''

# PARSE EACH JPEG AND CREATE CHARACTER DATASET
# Initialize lists for storing images and labels
X = []

# Define the number of samples to display per file
num_samples = 5

# Process each image
for filename in os.listdir(directory):
    if filename.endswith(".jpeg"):
        filepath = os.path.join(directory, filename)
        print(f"Processing {filename}...")

        # Read the image in grayscale
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # Apply specific transformations based on the filename
        if filename == 'image_page_0_1.jpeg':
            image = cv2.rotate(image, cv2.ROTATE_180)
        if filename == 'image_page_3_1.jpeg':
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) 
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) 
        if filename == 'image_page_4_1.jpeg':
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  
        if filename == 'image_page_5_1.jpeg':
            image = cv2.rotate(image, cv2.ROTATE_180) 
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) 
        if filename == 'image_page_6_1.jpeg':
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)     
        if filename == 'image_page_7_1.jpeg':
            #image = cv2.rotate(image, cv2.ROTATE_180)
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)     
        if filename == 'image_page_8_1.jpeg': 
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) 
        if filename == 'image_page_9_1.jpeg':
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) 
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) 
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  
            
                  

        # Apply adaptive thresholding to the image to create a binary image
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to remove noise by keeping only those with an area greater than 100
        contours = [ctr for ctr in contours if cv2.contourArea(ctr) > 100]

        # Sort contours from left to right and top to bottom based on their bounding rectangle
        contours = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[0], cv2.boundingRect(ctr)[1]))

        # Initialize a list to store the expanded character images
        expanded_characters = []

        # Define padding sizes for specific files
        padding_sizes = {
            'image_page_0_1.jpeg': 15, #12
            'image_page_1_1.jpeg': 16, #12
            'image_page_2_1.jpeg': 17, #14
            'image_page_3_1.jpeg': 20, #16
            'image_page_4_1.jpeg': 21, #18
            'image_page_5_1.jpeg': 23, #20
             # [0,2,3,4] | {'A': 0, 'C': 2, 'D': 3, 'E': 4} | CAN NOT FIND 'B': 1
             # Predicted Class Counts: {A:35, B:0, C:4, D:1120, E:6}
            'image_page_6_1.jpeg': 7, #7
            'image_page_7_1.jpeg': 7, #7
            'image_page_8_1.jpeg': 20, #20
            'image_page_9_1.jpeg': 21, #21
        }

        # Get the padding size for the current file, default to 14 if not specified
        padding = padding_sizes.get(filename, 14)

        # Loop through each contour to process and expand the capture window
        for ctr in contours:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(ctr)
            
            # Adjust the bounding rectangle to include padding, ensuring it stays within image bounds
            x = max(0, x - padding)
            y = max(0, y - padding)
            # Define width and height adjustments for specific files
            width_height_adjustments = {
                'image_page_0_1.jpeg': (3 * padding, 6 * padding),
                'image_page_1_1.jpeg': (3 * padding, 6 * padding),
                'image_page_2_1.jpeg': (3 * padding, 4 * padding),
                'image_page_3_1.jpeg': (4 * padding, 4 * padding),
                'image_page_4_1.jpeg': (3 * padding, 5 * padding),
                'image_page_5_1.jpeg': (3 * padding, 4 * padding),

                #'image_page_6_1.jpeg': (3 * padding, 4 * padding),
                #'image_page_7_1.jpeg': (3 * padding, 4 * padding),
                #'image_page_8_1.jpeg': (3 * padding, 3 * padding),
                'image_page_9_1.jpeg': (3 * padding, 6 * padding),
            }

            # Get the width and height adjustments for the current file, default to (2 * padding, 2 * padding) if not specified
            width_adjustment, height_adjustment = width_height_adjustments.get(filename, (3 * padding, 3 * padding))

            # Adjust the bounding rectangle to include padding and custom width/height adjustments, ensuring it stays within image bounds
            w = min(image.shape[1] - x, w + width_adjustment)
            h = min(image.shape[0] - y, h + height_adjustment)
            #w = min(image.shape[1] - x, w + (3 * padding))
            #h = min(image.shape[0] - y, h + (3 * padding))

            
            # Extract the region of interest (ROI) from the image using the adjusted bounding rectangle
            roi = image[y:y+h, x:x+w]
            
            # Resize the ROI to 28x28 pixels to maintain consistency with model input requirements
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Append the processed ROI to the list of expanded characters
            expanded_characters.append(roi)
            
            # Flatten the ROI and store it in the X list for model input
            X.append(roi.flatten())
            
        
        # Sample a few images to display
        sampled_indices = random.sample(range(len(expanded_characters)), min(num_samples, len(expanded_characters)))
        sampled_images = [expanded_characters[i] for i in sampled_indices]

        # Display sampled characters for verification
        print(f"Displaying {num_samples} sampled characters from {filename}.")
        fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
        for i, ax in enumerate(axes):
            if i < len(sampled_images):
                ax.imshow(sampled_images[i], cmap='gray')
                ax.set_title(f'Sample: {i+1}')
                ax.set_xticks([])
                ax.set_yticks([])
        plt.show()


'''
PREPROCESS DATA FOR MODEL USE
'''
y_test = char_order
print("Filtered Y Test | Labels:", np.unique(y_test))

# Remap the labels A,B,C,D,E to 0-4 for the test set
y_test = np.array([char_labels[label] for label in y_test])
print("NEW Y Labels:", y_test)

try:
    # Reshape X to have the correct dimensions (number of samples, height, width, channels)
    X = np.array(X, dtype=np.float32).reshape(-1, 28, 28)  # Assuming the images are 28x28 pixels with 1 channel
    print("Shape after reshaping:", X.shape)

    # Normalize
    X = X / 255.0  # Normalize pixel values to the range [0, 1]
    print("Shape after normalization:", X.shape)

    # Add batch and channel dimensions if necessary
    X = np.expand_dims(X, axis=-1)
    print("Shape after ensuring dimensions:", X.shape)

    # Resize images to (32, 32, 3)
    X = tf.image.resize(X, (32, 32))
    print("x_test re-size shape:", X.shape)

   # Convert to 3 channels
    X = tf.image.grayscale_to_rgb(X)
    print("Shape after converting to RGB:", X.shape)
    print("--------------------\n")

except Exception as e:
    print("An error occurred:", e)

# Load model and predict
updated_model = tf.keras.models.load_model('C:/ML2/ML2_HOMEWORK/mnist_EfficientNetB0_UPDATED.h5')
updated_model.summary()

# Predict the categories for the images
predictions = updated_model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)
print("Predicted Classes:", np.unique(predicted_classes))

# Count the number of predictions for each class
unique, counts = np.unique(predicted_classes, return_counts=True)
predicted_class_counts = dict(zip(unique, counts))
print("Predicted Class Counts:", predicted_class_counts)
# To run Tensorboard, use the following command in your terminal:
# tensorboard --logdir logs/fit














'''
FINE TUNE
'''

# Get predicted classes from the existing model
predicted_classes = np.argmax(predictions, axis=1)
print("PRedicted Class Check:",predicted_classes)

# Compute class weights dynamically from discovered classes
class_labels = np.unique(predicted_classes)  # Found classes
classes = np.arange(len(char_labels))
print("Classes:",classes) # [0 1 2 3 4]
# Compute class weights dynamically from the full range of classes
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.concatenate([predicted_classes, np.arange(len(char_labels))]))
print("Class Weights:", class_weights)

# Convert to dictionary
class_weights_dict = {i: class_weights[i] for i in classes}
#print("Computed Class Weights:", class_weights_dict)

# APPLY MATRIX MULTIPLCATION TO APPLY WEIGHTS
adjusted_predictions = np.dot(predictions, np.diag([class_weights_dict[i] for i in np.arange(len(char_labels))]))
#print("Adjusted Predictions:", adjusted_predictions)

# Get new predicted classes after weight adjustment
final_predicted_classes = np.argmax(adjusted_predictions, axis=1)
print("Final Predicted Classes:", np.unique(final_predicted_classes))

# Count the number of predictions for each class
unique_final, counts_final = np.unique(final_predicted_classes, return_counts=True)
final_predicted_class_counts = dict(zip(unique_final, counts_final))
print("Final Predicted Class Counts:", final_predicted_class_counts)


'''
MANUALLY REVIEW PREDICTIONS TO ADD LABELS
'''
# Map the predicted class indices back to their original labels
final_predicted_class_labels = {char_order[key]: value for key, value in final_predicted_class_counts.items()}
print("Final Predicted Class Labels:", final_predicted_class_labels)

# Plot the final predicted class counts
plt.figure(figsize=(10, 6))
plt.bar(final_predicted_class_labels.keys(), final_predicted_class_labels.values(), color='skyblue')
plt.xlabel('Character')
plt.ylabel('Count')
plt.title('Final Predicted Class Counts')
plt.show()