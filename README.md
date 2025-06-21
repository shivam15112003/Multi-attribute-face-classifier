# 🚀 Multi-Task Image Classification & Regression with Transfer Learning and Fine-Tuning

## Project Overview

This project performs multi-task human image analysis using deep learning models to predict:
- Age (Regression)
- Nationality (Classification)
- Dress Code (Classification)

The project uses transfer learning with pretrained CNN models (VGG16 and MobileNetV2) followed by fine-tuning for optimal performance.

---

## Dataset Structure

The dataset must be structured as follows:

```
dataset/
  ├── age/
  │     ├── 10/
  │     ├── 20/
  │     └── ...
  ├── nationality/
  │     ├── African/
  │     ├── American/
  │     ├── Australian/
  │     ├── Brazilian/
  │     └── Indian/
  └── dresscode/
        ├── Casual/
        ├── Formal/
        └── Semi-formal/
```

---

## Preprocessing

- Image resizing: 224x224 pixels
- Normalization: Scale pixel values between 0 and 1
- Data augmentation: horizontal flipping, brightness, and contrast adjustments
- Validation split: 20%

---

## Model Architectures

### Age Prediction Model (Regression)

- Base Model: VGG16 (ImageNet pretrained)
- Custom Layers: GAP, Dense(128), Dropout(30%), Output(Dense 1, Linear)
- Loss: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)

### Nationality Classification Model (Classification)

- Base Model: MobileNetV2 (ImageNet pretrained)
- Custom Layers: GAP, BatchNorm, Dense(256), Dropout(50%), Output(Softmax with 5 classes)
- Loss: Categorical Crossentropy
- Metric: Accuracy

### Dress Code Classification Model (Classification)

- Same architecture as Nationality Classification with 3 output classes.

---

## Training Pipeline

1. **Transfer Learning Phase**: 
   - Freeze pretrained layers
   - Train only the new dense layers
2. **Fine-Tuning Phase**: 
   - Unfreeze deeper layers (last 5 for VGG16, last 10 for MobileNetV2)
   - Train with a lower learning rate

### Training Details

- Optimizer: Adam
- Learning Rates: 0.0001 (transfer learning), 0.00001 (fine-tuning)
- Batch Size: 32
- Epochs: 10 (transfer learning) + 5 (fine-tuning)
- Callbacks: EarlyStopping, ReduceLROnPlateau

---

## Output

The models are saved as:
- `models/age_model.h5`
- `models/nationality_model.h5`
- `models/dresscode_model.h5`

---

## Future Scope

- Integration into Tkinter GUI for real-time predictions.
- Expand dataset to more nationalities and dress codes.
- Add emotion detection module.

---

## Keywords

VGG16, MobileNetV2, CNN, Deep Learning, Transfer Learning, Fine-Tuning, Regression, Classification, Image Processing.

