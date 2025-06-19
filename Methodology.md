# Methodology: Multi-Attribute Face Classification System

This methodology outlines the workflow and technical approach used in building the Multi-Attribute Face Classifier GUI application.

---

## 1. Objective

To develop a deep learning-based desktop application that can accurately predict multiple attributes from a facial image:

* Age (10–60 years)
* Nationality
* Emotion
* Dress code (for African and Indian nationalities)

---

## 2. Data Collection & Preparation

### 2.1 Datasets Used

* **Age Dataset:** A curated set of facial images labeled by age group (ages 6–100, but filtered to 10–60).
* **Nationality Dataset:** Images categorized into five nationalities — African, American, Australian, Brazilian, Indian.
* **Emotion Dataset:** Utilized by DeepFace API for emotion recognition.
* **Dress Code Dataset:** Images labeled as Casual, Formal, or Semi-formal.

### 2.2 Preprocessing Techniques

* **Resizing** all images to 224x224 pixels.
* **Histogram Equalization** for better contrast.
* **Data Augmentation**:

  * Random horizontal flipping
  * Brightness and contrast variation
* **Normalization** to scale pixel values between 0 and 1.

---

## 3. Model Architecture

### 3.1 Age Estimation

* **Base Model:** VGG16 (pre-trained on ImageNet)
* **Modification:**

  * Global Average Pooling
  * Dense Layer (128 neurons)
  * Output Layer: Single neuron with linear activation
* **Loss Function:** Mean Squared Error (MSE)

### 3.2 Nationality and Dress Code Classification

* **Base Model:** MobileNetV2
* **Modifications:**

  * Global Average Pooling
  * Batch Normalization
  * Dense (256 neurons, ReLU)
  * Output Layer: Softmax (5 for nationality, 3 for dress code)
* **Loss Function:** Categorical Crossentropy

### 3.3 Emotion Recognition

* Handled using **DeepFace** library with built-in facial expression analysis.

---

## 4. Training Details

* **Epochs:** 10–15 (tuned per model)
* **Batch Size:** 32
* **Optimizer:** Adam with reduced learning rate (`0.00005`)
* **Validation Split:** 10–20% of dataset
* **Regularization:** Dropout & Early stopping (during training phase)

---

## 5. Integration with GUI

* Developed a desktop GUI using **Tkinter**.
* GUI allows users to:

  * Upload an image
  * Preview it
  * Get real-time predictions for age, nationality, emotion, and dress code
* Ensures predictions are only made for age between 10 and 60.

---

## 6. Evaluation Metrics

* **Age:** Mean Absolute Error (MAE)
* **Nationality/Dress Code:** Classification Accuracy
* **Emotion:** Reported by DeepFace

---

## 7. Constraints and Conditions

* Predictions limited to images with a clear front-facing face.
* Dress code prediction applies only to users from African or Indian backgrounds.

---

## 8. Conclusion

The methodology successfully combines modern CNN architectures with preprocessing and GUI integration, offering a practical and accurate multi-attribute classification system from static images.

---

**Keywords:** VGG16, MobileNetV2, CNN, Emotion Detection, Deep Learning, GUI, Tkinter, DeepFace
