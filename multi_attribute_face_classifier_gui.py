import tensorflow as tf
from keras.applications import VGG16, MobileNetV2
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from deepface import DeepFace
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
import os

# ---------------------------- Model Definitions ---------------------------- #
def build_age_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='mse', metrics=['mae'])
    return model

def build_classification_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------- Load Models ---------------------------- #
age_model = build_age_model()
nationality_model = build_classification_model(5)
dresscode_model = build_classification_model(3)

# Load trained weights
age_model.load_weights('models/age_model.h5')
nationality_model.load_weights('models/nationality_model.h5')
dresscode_model.load_weights('models/dresscode_model.h5')

# ---------------------------- Utility Functions ---------------------------- #
def extract_class_name(predictions, classes):
    predicted_indices = predictions.argmax(axis=1)
    return [classes[idx] for idx in predicted_indices]

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = tf.image.resize(img, (224, 224)) / 255.0
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return np.expand_dims(img, axis=0)

def plot_training(history, title):
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    if val_acc: plt.plot(val_acc, label='Validation Accuracy')
    plt.title(f'{title} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    if val_loss: plt.plot(val_loss, label='Validation Loss')
    plt.title(f'{title} Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ---------------------------- Main Analysis Function ---------------------------- #
def analyze_image(img_path):
    img = preprocess_image(img_path)

    # Predict age
    age = int(age_model.predict(img)[0][0])

    if not (10 <= age <= 60):
        return f"Age out of range (10-60). Detected: {age}", []

    results = [f"Predicted Age: {age}"]

    # Predict nationality
    nat_preds = nationality_model.predict(img)
    nat_class = extract_class_name(nat_preds, nationality_classes)[0]
    results.append(f"Predicted Nationality: {nat_class}")

    # Predict emotion
    face_analysis = DeepFace.analyze(img_path=img_path, actions=['emotion'])[0]
    results.append(f"Predicted Emotion: {face_analysis['dominant_emotion']}")

    # Predict dress code if required
    if nat_class in ['African', 'Indian']:
        dress_preds = dresscode_model.predict(img)
        dress_class = extract_class_name(dress_preds, dresscode_classes)[0]
        results.append(f"Predicted Dress Code: {dress_class}")

    return None, results

# ---------------------------- GUI Setup ---------------------------- #
window = tk.Tk()
window.title("Person Classification App")
window.geometry("600x600")

nationality_classes = ['African', 'American', 'Austrialian', 'Brazilian', 'Indian']
dresscode_classes = ['Casual', 'Formal', 'Semi-formal']

def clear_messages():
    for widget in frame_display.winfo_children():
        widget.destroy()

def browse_files():
    filename = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if not filename:
        return

    clear_messages()

    # Show preview
    img = Image.open(filename)
    img = img.resize((200, 200))
    photo = ImageTk.PhotoImage(img)
    label_img = tk.Label(frame_display, image=photo)
    label_img.image = photo
    label_img.pack()

    # Run analysis
    error, results = analyze_image(filename)
    if error:
        tk.Label(frame_display, text=error, fg="red").pack()
    else:
        for res in results:
            tk.Label(frame_display, text=res).pack()

frame_top = tk.Frame(window)
frame_top.pack(pady=20)

btn_upload = tk.Button(frame_top, text="Upload Image", command=browse_files)
btn_upload.pack()

frame_display = tk.Frame(window)
frame_display.pack(pady=10)

window.mainloop()
