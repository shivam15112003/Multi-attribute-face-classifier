# 🚀 Multi-Attribute Face Classifier GUI

This project is a **Tkinter-based desktop application** that predicts multiple attributes from a person's facial image. It leverages **deep learning models** for classification tasks such as:

* **Age Estimation** (10–60 years)
* **Nationality Prediction**
* **Emotion Recognition**
* **Dress Code Classification** (for select nationalities)

## 🔧 Features

* Interactive GUI for selecting images
* Pre-trained models using **VGG16** and **MobileNetV2**
* Real-time predictions with image preview
* Enhanced accuracy with advanced **image preprocessing** and **regularization**

## 📂 Project Structure

```
├── models/
│   ├── age_model.h5
│   ├── nationality_model.h5
│   └── dresscode_model.h5
├── person_classification_gui.py
├── README.md
```

## 🧠 Requirements

* Python 3.7+
* TensorFlow
* Keras
* OpenCV
* DeepFace
* Pillow
* Matplotlib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

1. Place your trained model files (`.h5`) in the `models/` directory.
2. Run the GUI:

```bash
python person_classification_gui.py
```

3. Click **Upload Image** and select a file to classify.

## 📈 Training Visualization

You can visualize training/validation accuracy and loss using the built-in `plot_training(history, title)` function if you decide to re-train models.

## ✅ Output Example

```
Predicted Age: 28
Predicted Nationality: Indian
Predicted Emotion: happy
Predicted Dress Code: Semi-formal
```

## 📌 Notes

* Age predictions are constrained to \[10–60] years. If outside this range, predictions are skipped.
* Dress code prediction is only shown for Indian or African individuals.



Made with ❤️ using TensorFlow and Deep Learning.
