import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks, optimizers, metrics
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pathlib
import matplotlib.pyplot as plt

# ======================
# Configuration
# ======================
IMG_SIZE = (384, 384)  # Optimal size for EfficientNetV2
BATCH_SIZE = 64
EPOCHS = 100
BASE_DIR = pathlib.Path("dataset2")
CLASS_NAMES = ["tail pass", "tail fail", "flake pass", "flake fail"]  # Maintain folder name order

# ======================
# Data Pipeline
# ======================
# Load datasets from separate directories
train_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode='int',
    class_names=CLASS_NAMES  # Ensure consistent label mapping
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR / "validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode='int',
    class_names=CLASS_NAMES
)

# ======================
# Class Weight Calculation
# ======================
def get_class_weights(dataset):
    labels = []
    for _, batch_labels in dataset.unbatch():
        labels.extend(batch_labels.numpy())
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(labels), 
        y=labels
    )
    return dict(enumerate(class_weights))

class_weights = get_class_weights(train_ds)

# ======================
# Advanced Data Augmentation
# ======================
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.GaussianNoise(0.01),
    layers.RandomTranslation(0.1, 0.1),
])

# ======================
# Optimized Model Architecture (EfficientNetV2)
# ======================
def build_model():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    
    # EfficientNetV2 preprocessing
    x = applications.efficientnet_v2.preprocess_input(x)
    
    # Base model
    base_model = applications.EfficientNetV2L(
        include_top=False,
        weights='imagenet',
        input_tensor=x,
        include_preprocessing=False
    )
    base_model.trainable = False

    # Custom head with regularization
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(1024, activation='swish')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

    return models.Model(inputs, outputs)

model = build_model()

# ======================
# Advanced Training Setup
# ======================
model.compile(
    optimizer=optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=[
        'accuracy',
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc')
    ]
)

# ======================
# Enhanced Callbacks
# ======================
callbacks = [
    callbacks.ModelCheckpoint(
        "best_model.keras",
        save_best_only=True,
        monitor='val_auc',
        mode='max',
        save_weights_only=False
    ),
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
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
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ======================
# Fine-tuning Phase
# ======================
model = models.load_model("best_model.keras")
model.layers[2].trainable = True  # Unfreeze base model

# Set lower learning rate for fine-tuning
model.compile(
    optimizer=optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=model.metrics
)

history_fine = model.fit(
    train_ds,
    epochs=EPOCHS//2,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ======================
# Model Evaluation
# ======================
def plot_performance(history):
    plt.figure(figsize=(18, 6))
    
    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    
    # AUC
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Train')
    plt.plot(history.history['val_auc'], label='Validation')
    plt.title('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_performance(history)
plot_performance(history_fine)

# ======================
# Optimized Inference
# ======================
def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = applications.efficientnet_v2.preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[tf.argmax(predictions[0]).numpy()]
    confidence = tf.reduce_max(predictions[0]).numpy()
    
    return predicted_class, confidence

# Save final model
model.save("plastic_classifier_final.keras")
