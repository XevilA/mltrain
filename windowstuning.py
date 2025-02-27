import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks, optimizers, metrics
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# Configuration
# ======================
IMG_SIZE = (384, 384)
BATCH_SIZE = 32
EPOCHS = 100
BASE_DIR = pathlib.Path(r"C:\Users\ADMIN\Desktop\defect_training\dataset2")
CLASS_NAMES = ["tail_pass", "tail_fail", "flake_pass", "flake_fail"]

# ======================
# Mixed Precision Setup
# ======================
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ======================
# Enhanced Data Pipeline
# ======================
def create_dataset(subset):
    ds = tf.keras.utils.image_dataset_from_directory(
        BASE_DIR / subset,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(subset == "train"),
        label_mode='int',
        class_names=CLASS_NAMES,
        seed=42
    )
    return ds.map(
        lambda x, y: (applications.efficientnet_v2.preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).cache().prefetch(tf.data.AUTOTUNE)

train_ds = create_dataset("train")
val_ds = create_dataset("validation")

# ======================
# Class Weight Calculation
# ======================
def get_class_weights(dataset):
    labels = []
    for _, batch_labels in dataset.unbatch():
        labels.append(batch_labels.numpy())
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(class_weights))

class_weights = get_class_weights(train_ds)

# ======================
# Advanced Augmentation
# ======================
def augment():
    return models.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
        layers.GaussianNoise(0.02),
        layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")

# ======================
# Model Architecture
# ======================
def build_model():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = augment()(inputs)
    
    # EfficientNetV2L with Imagenet weights
    base_model = applications.EfficientNetV2L(
        include_top=False,
        weights='imagenet',
        input_tensor=x
    )
    base_model.trainable = False

    # Advanced Head Architecture
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(1024, activation='swish')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='swish')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation='softmax', dtype='float32')(x)

    return models.Model(inputs, outputs)

model = build_model()

# ======================
# Optimized Training Setup
# ======================
initial_learning_rate = 1e-4
lr_schedule = optimizers.schedules.CosineDecay(
    initial_learning_rate, 
    EPOCHS * len(train_ds)
)  # Fixed closing parenthesis

model.compile(
    optimizer=optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
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
callbacks_list = [
    callbacks.ModelCheckpoint(
        "best_model.keras",
        save_best_only=True,
        monitor='val_auc',
        mode='max',
        save_weights_only=False
    ),
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ),
    callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1
    ),
    callbacks.BackupAndRestore("backup"),
    callbacks.TerminateOnNaN()
]

# ======================
# Training Execution
# ======================
try:
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )
except Exception as e:
    print(f"Training failed: {str(e)}")
    exit(1)

# ======================
# Fine-tuning Phase
# ======================
try:
    model = models.load_model("best_model.keras")
    model.layers[2].trainable = True
    
    # Unfreeze top 50 layers
    for layer in model.layers[2].layers[-50:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=optimizers.AdamW(learning_rate=1e-6, weight_decay=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=model.metrics
    )

    history_fine = model.fit(
        train_ds,
        epochs=EPOCHS//2,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )
except Exception as e:
    print(f"Fine-tuning failed: {str(e)}")
    exit(1)

# ======================
# Evaluation & Visualization
# ======================
def plot_metrics(history):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Train')
    plt.plot(history.history['val_auc'], label='Validation')
    plt.title('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_metrics(history)
model.save("final_production_model.keras")

print("Training completed successfully!")
