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
Clone the repository and run the following command:
```bash
python train_model.py # Needs to be validated as of 2/28/25
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
- **Callout:** this model is built for five class detection, hence the outputs are set to 5.
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
- Only the final layers were retrained to specialize the model for the new dataset, reducing overfitting

**Final Test Accuracy Achieved: 92.6%**

---

## Handwritten Letters Dataset
A new dataset consisting of characters **A, B, C, D, and E** was introduced.

### Preprocessing:
- Extracted images from a **PDF file**.
- Applied **adaptive thresholding and contour detection**.
- Resized images to **28x28** and converted to **grayscale RGB**.
- **Fine-tuned the model** using this new dataset.

**Current Challenge:**
- The model struggles with detecting the letter **B**.
- Work is ongoing to **adjust weights** and **balance the dataset**.

---

## Results & Performance
### Final Predictions on Handwritten Letters:
| Letter | Predicted Count |
|--------|----------------|
| A      | 35             |
| B      | 0              |
| C      | 4              |
| D      | 1120           |
| E      | 6              |

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

## Next Steps
- **Improve detection of letter B** in the new dataset.
- **Optimize data augmentation** for letter recognition.
- **Deploy the model** for real-world image classification tasks.
- **Experiment with additional architectures** (e.g., ResNet, VGG16).

---

## Contributor
- **Lani Lewis** 

For inquiries, reach out at [lani.k.lewis2@gmail.com](mailto:lani.k.lewis2@gmail.com)

