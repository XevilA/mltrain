"""
Developer Profile:
-------------------------
Name: Tirawat Nantamas
Position: CEO and Founder of Dotmini Software & Defense
Skills: HTML, CSS, JavaScript, PHP, MySQL, Adobe XD, Figma, UX/UI Design, Python, Go, Rust, Linux (RHEL9), macOS, Firebase Hosting, TensorFlow, Keras, Machine Learning, AI, Image Classification, Deep Learning
Tools & Technologies: TensorFlow, Keras, OpenCV, Scikit-learn, Seaborn, Matplotlib, PyTorch, Jupyter Notebooks, GitHub, VSCode, Zed Editor, Git
Projects: Developing AI/ML models for image classification, deep learning model training pipelines, and modern UX/UI applications.
GitHub: https://github.com/Tirawat-Nantamas

Copyright:
-----------
Copyright (c) 2025 Tirawat Nantamas. All rights reserved.
This code is licensed under the MIT License. See the LICENSE file for more details.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks, optimizers, metrics, mixed_precision
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras_tuner as kt
import os

# ======================
# Configuration
# ======================
IMG_SIZE = (384, 384)  # Increased for higher resolution
BATCH_SIZE = 64
EPOCHS = 100
DATA_DIR = "PATH"
CLASS_NAMES = ["flake_pass", "flake_fail", "tail_pass", "tail_fail"]

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE
mixed_precision.set_global_policy('mixed_float16')  # Faster training on modern GPUs

# ======================
# Advanced Data Pipeline
# ======================
def create_dataset(data_dir, validation_split=0.2):
    file_paths = list(data_dir.glob('*/*.jpg'))
    labels = [CLASS_NAMES.index(path.parent.name) for path in file_paths]
    
    # Stratified split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=validation_split, stratify=labels, random_state=42
    )

    def process_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img, label

    def augment(img, label):
        # Advanced augmentation
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_hue(img, 0.1)
        img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE[0]+32, IMG_SIZE[1]+32)
        img = tf.image.random_crop(img, size=[*IMG_SIZE, 3])
        return img, label

    def prepare(img, label, augment=False):
        img = tf.image.resize(img, IMG_SIZE)
        img = applications.efficientnet.preprocess_input(img)
        if augment:
            img = augment(img, label)
        return img, label

    # Training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(1024, reshuffle_each_iteration=True)
    train_ds = train_ds.map(process_image, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(lambda x,y: prepare(x,y, augment=True), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # Validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(process_image, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x,y: prepare(x,y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_ds, val_ds

train_ds, val_ds = create_dataset(DATA_DIR)

# ======================
# Hyperparameter Tuning Setup
# ======================
def model_builder(hp):
    base_model = applications.EfficientNetV2L(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3),
        pooling='avg'
    )
    
    # Tunable hyperparameters
    base_model.trainable = hp.Boolean('freeze_base', default=True)
    dropout_rate = hp.Float('dropout', 0.3, 0.7, step=0.1)
    learning_rate = hp.Float('lr', 1e-5, 1e-3, sampling='log')
    
    # Model architecture
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs)
    x = layers.Dense(hp.Int('units', 512, 1024, step=128), activation='swish')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.BayesianOptimization(
    model_builder,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=2,
    directory='tuning',
    project_name='plastic_classification'
)

# ======================
# Advanced Training Setup
# ======================
def get_callbacks():
    return [
        callbacks.ModelCheckpoint(
            "best_model.keras",
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.95 if epoch > 10 else lr
        ),
        callbacks.TensorBoard(
            log_dir='logs',
            profile_batch=0
        ),
        callbacks.BackupAndRestore("backup")
    ]

# ======================
# Training Execution
# ======================
# Phase 1: Hyperparameter search
tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=get_callbacks()
)

# Get best hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]

# Phase 2: Full training with best params
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=get_callbacks()
)

# ======================
# Precision Optimization
# ======================
# Convert to quantized model for efficiency (optional)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

with open('plastic_classifier_quantized.tflite', 'wb') as f:
    f.write(quantized_model)

# ======================
# Advanced Evaluation
# ======================
def evaluate_model(model, dataset):
    y_true = []
    y_pred = []
    
    for images, labels in dataset:
        y_true.extend(labels.numpy())
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
    
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=CLASS_NAMES)
    
evaluate_model(model, val_ds)

# ======================
# Optimized Inference
# ======================
@tf.function(experimental_compile=True)  # XLA compilation for faster inference
def predict_batch(images):
    return model(images, training=False)

def optimized_predict(image_paths):
    batch = []
    for path in image_paths:
        img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = applications.efficientnet.preprocess_input(img)
        batch.append(img)
    
    batch = tf.convert_to_tensor(batch)
    predictions = predict_batch(batch)
    return tf.argmax(predictions, axis=1).numpy()
