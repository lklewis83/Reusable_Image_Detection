# EfficientNetB0-based Reusable Image Detection Model

## Overview

This project is an image detection model built using EfficientNetB0 and the MNIST dataset. The model was trained to recognize handwritten digits (0-4 for training and 5-9 for testing). After training and testing, the model was further fine-tuned using transfer learning against an unseen, manually created dataset of alpha characters (A, B, C, D, E). The implementation took approximately 6-7 days to understand, build, train, test, and fine-tune.

### Features:

- **Transfer Learning** using EfficientNetB0
- **K-Fold Cross-Validation** to evaluate performance
- **Data Augmentation** for improved accuracy
- **Fine-Tuning & Reuse** with new datasets
- **Support for additional datasets** (e.g., handwritten letters: A, B, C, D, E)

---

## Dataset

### MNIST Digits:

- **Training Set:** Digits 0-4
- **Testing Set:** Digits 5-9

### Additional Dataset:

- Handwritten **A, B, C, D, E** characters (converted from PDF images)

---

## Installation & Setup

### Requirements:

Make sure you have the required libraries installed:

```bash
pip install tensorflow numpy scipy scikit-learn matplotlib opencv-python imbalanced-learn PyMuPDF
```

### Running the Model:

Clone the repository and run the following commands based on your needs:

#### Train the original model (digits 0-4):

```bash
python VisualNN.py
```

#### Fine-tune on unseen data (digits 5-9):

```bash
python Reuse_VisualNN_test.py
```

#### Transfer learning for character classification (A, B, C, D, E):

```bash
python Reuse_VisualNN.py
```

To fine-tune with new datasets, modify `fine_tune_model.py` accordingly.

---

## Model Architecture

### Initial Model:

A simple feedforward model was used for initial training:

```python
def create_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(5, activation='softmax')(x)  # 5 classes
    return Model(inputs=inputs, outputs=outputs)
```

### Training & Cross-Validation:

- **Callout:** this model is built for five-class detection, hence the outputs are set to 5.
- **5-Fold Cross-Validation** was implemented.
- The best model (highest accuracy) was saved as `mnist_EfficientNetB0.h5`.

**Best Accuracy Achieved: 99.2%**

---

## Fine-Tuning & Reuse

### Reusing the Model for Testing:

```python
base_model = tf.keras.models.load_model('mnist_EfficientNetB0.h5')
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)
```

- **Data Augmentation** was applied to improve test accuracy.
- **Early Stopping & Learning Rate Scheduling** were used for better training.

#### Freezing the Top Layers:

- When fine-tuning, the top layers of the pre-trained EfficientNetB0 model were frozen. This means these layers retained their pre-trained weights and were not updated during training.
- Freezing prevents the model from forgetting previously learned features while adapting to new data.
- Only the final layers were retrained to specialize the model for the new dataset, reducing overfitting.

**Final Test Accuracy Achieved: 92.6%**

---

## Handwritten Letters Dataset

A new dataset consisting of characters **A, B, C, D, and E** was introduced.

### Preprocessing:

- Extracted images from a **PDF file**.
- Applied **adaptive thresholding and contour detection**.
- Resized images to **28x28** and converted to **grayscale RGB**.
- **Fine-tuned the model** using this new dataset.

### Model Adjustments for Class Imbalance:

- **Implemented Class Weights** to compensate for imbalanced class distribution.
- **Used Matrix Multiplication** to improve feature extraction and classification accuracy.
- **Focused on Dataset Balancing**: If additional time were available, the main focus would be on refining the image capturing process to ensure a more evenly distributed dataset across all classes.

---

### Final Predictions on Handwritten Letters:

| Letter | Predicted Count |
| ------ | --------------- |
| A      | 35              |
| B      | 0               |
| C      | 4               |
| D      | 1120            |
| E      | 6               |

### Addressing Class B Detection Issue

To handle the issue of class B not being detected at all, the following fine-tuning approach was applied:

1. **Computed Class Weights Dynamically:**

   - Retrieved predicted classes from the model.
   - Determined the discovered classes and ensured the full range was accounted for.
   - Computed class weights dynamically to balance the detected and missing classes.

2. **Applied Matrix Multiplication for Weight Adjustment:**

   - Adjusted predictions by multiplying them with a diagonal matrix of computed class weights.
   - This rebalanced the model’s bias towards the missing class B.

3. **Final Prediction Adjustments:**

   - Retrieved new predicted classes after weight adjustments.
   - Counted the final occurrences of each class to verify if class B was now being recognized.
   - Plotted the updated distribution of predictions to confirm improvements.

### Final Class Distribution After Adjustments:
- **Predicted Classes Before Adjustment:** `[3 3 3 ... 3 3 3]`
- **Computed Class Weights:** `[78.0, 234.0, 58.5, 0.202, 39.0]`
- **Final Predicted Classes After Weight Adjustment:** `[0, 1, 2, 3, 4]`
- **Final Predicted Class Counts:** `{'A': 72, 'B': 1, 'C': 6, 'D': 1075, 'E': 11}`

This confirms that class B, which was previously undetected, now has at least one recognized instance after applying class weight adjustments and matrix multiplication to rebalance the prediction distribution.

This method ensured a more balanced prediction outcome while addressing model bias toward overrepresented classes.

### Confusion Matrix on Test Dataset:

**Final Test Accuracy Achieved: 92.6%**

```plaintext
[[5690  111  318  113   81]
 [ 166 6621    3   86    0]
 [ 135    8 6711   41  398]
 [ 308   61   86 6285   85]
 [ 142   13  235  145 6423]]
```

---

## Files Inside This Branch

### Model Scripts

- **VisualNN.py** - Original model trained on digits 0-4.
- **Reuse\_VisualNN\_test.py** - Fine-tuned original model tested on unseen data (digits 5-9).
- **Reuse\_VisualNN.py** - Transfer learning model trained to classify five distinct character groups (A, B, C, D, E).

### Reusable Keras Models

- **mnist\_EfficientNetB0.h5** - Initial trained Keras model on MNIST (digits 0-4).
- **mnist\_EfficientNetB0\_UPDATED.h5** - Fine-tuned Keras model with transfer learning applied.

---

## Next Steps

- **Improve detection of letter B** in the new dataset.
- **Optimize data augmentation** for letter recognition.
- **Deploy the model** for real-world image classification tasks.
- **Experiment with additional architectures** (e.g., ResNet, VGG16).
- **Refine the dataset collection process** to ensure a more balanced representation of all classes.

---

## Contributor

- **Lani Lewis**

For inquiries, reach out at [lani.k.lewis2@gmail.com](mailto\:lani.k.lewis2@gmail.com)
