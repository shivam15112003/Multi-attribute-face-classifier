import tensorflow as tf
from keras.applications import VGG16, MobileNetV2
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os

# ---------------------------- Dataset Paths ---------------------------- #
# You need to adjust these directories according to your dataset structure
AGE_DATA_DIR = 'dataset/age'
NATIONALITY_DATA_DIR = 'dataset/nationality'
DRESSCODE_DATA_DIR = 'dataset/dresscode'

# ---------------------------- Model Definitions ---------------------------- #
def build_age_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
    return model, base_model

def build_classification_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base_model

# ---------------------------- Data Generators ---------------------------- #
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# AGE DATA (Regression task)
train_age = datagen.flow_from_directory(
    AGE_DATA_DIR,
    target_size=(224, 224),
    class_mode='sparse',
    batch_size=32,
    subset='training'
)
val_age = datagen.flow_from_directory(
    AGE_DATA_DIR,
    target_size=(224, 224),
    class_mode='sparse',
    batch_size=32,
    subset='validation'
)

# NATIONALITY DATA (Classification task)
train_nat = datagen.flow_from_directory(
    NATIONALITY_DATA_DIR,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    subset='training'
)
val_nat = datagen.flow_from_directory(
    NATIONALITY_DATA_DIR,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    subset='validation'
)

# DRESSCODE DATA (Classification task)
train_dress = datagen.flow_from_directory(
    DRESSCODE_DATA_DIR,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    subset='training'
)
val_dress = datagen.flow_from_directory(
    DRESSCODE_DATA_DIR,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    subset='validation'
)

# ---------------------------- Training ---------------------------- #

# 1️⃣ AGE Model
age_model, age_base = build_age_model()
callbacks = [EarlyStopping(patience=5), ReduceLROnPlateau(patience=3)]
age_model.fit(train_age, validation_data=val_age, epochs=10, callbacks=callbacks)

# Fine-tune last few layers
for layer in age_base.layers[-5:]:
    layer.trainable = True
age_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])
age_model.fit(train_age, validation_data=val_age, epochs=5, callbacks=callbacks)
age_model.save('models/age_model.h5')

# 2️⃣ NATIONALITY Model
nationality_model, nat_base = build_classification_model(num_classes=train_nat.num_classes)
nationality_model.fit(train_nat, validation_data=val_nat, epochs=10, callbacks=callbacks)

for layer in nat_base.layers[-10:]:
    layer.trainable = True
nationality_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
nationality_model.fit(train_nat, validation_data=val_nat, epochs=5, callbacks=callbacks)
nationality_model.save('models/nationality_model.h5')

# 3️⃣ DRESSCODE Model
dresscode_model, dress_base = build_classification_model(num_classes=train_dress.num_classes)
dresscode_model.fit(train_dress, validation_data=val_dress, epochs=10, callbacks=callbacks)

for layer in dress_base.layers[-10:]:
    layer.trainable = True
dresscode_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
dresscode_model.fit(train_dress, validation_data=val_dress, epochs=5, callbacks=callbacks)
dresscode_model.save('models/dresscode_model.h5')

print("All models trained and saved successfully!")