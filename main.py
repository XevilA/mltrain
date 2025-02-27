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

import os


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TrainClassificationModel:
    def __init__(self, data_dir, img_height=224, img_width=224, batch_size=32):
        """
        Initialize the train classification model.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the train images dataset
        img_height : int
            Height to resize images to
        img_width : int
            Width to resize images to
        batch_size : int
            Batch size for training
        """
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None
        
    def prepare_data(self, validation_split=0.2):
        """
        Prepare and augment the dataset for training.
        
        Parameters:
        -----------
        validation_split : float
            Fraction of data to use for validation
            
        Returns:
        --------
        train_ds, val_ds : tf.data.Dataset
            Training and validation datasets
        """
        print("Preparing data...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Load training data
        self.train_ds = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Load validation data
        self.val_ds = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Store class names
        self.class_names = list(self.train_ds.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        print(f"Dataset prepared with {len(self.class_names)} classes: {self.class_names}")
        print(f"Training samples: {self.train_ds.samples}")
        print(f"Validation samples: {self.val_ds.samples}")
        
        return self.train_ds, self.val_ds
    
    def build_model(self):
        """
        Build and compile the CNN model for train classification.
        
        Returns:
        --------
        model : tf.keras.Model
            Compiled model
        """
        print("Building model...")
        
        # Create a CNN model
        model = Sequential([
            # First convolutional layer
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(self.img_height, self.img_width, 3)),
            MaxPooling2D((2, 2)),
            
            # Second convolutional layer
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            
            # Third convolutional layer
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            
            # Fourth convolutional layer
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            
            # Flatten the output and feed it into dense layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),  # Add dropout for regularization
            Dense(self.num_classes, activation='softmax')  # Output layer with softmax activation
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, epochs=50):
        """
        Train the model on the prepared dataset.
        
        Parameters:
        -----------
        epochs : int
            Number of epochs to train
            
        Returns:
        --------
        history : tf.keras.callbacks.History
            Training history
        """
        if self.model is None:
            self.build_model()
            
        print("Training model...")
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
            ModelCheckpoint('best_train_model.h5', save_best_only=True)
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return self.history
    
    def evaluate_model(self):
        """
        Evaluate the model on the validation set and generate metrics.
        
        Returns:
        --------
        evaluation_metrics : dict
            Dictionary containing evaluation metrics
        """
        if self.model is None or self.history is None:
            print("Model needs to be trained first.")
            return None
            
        print("Evaluating model...")
        
        # Get predictions for validation data
        self.val_ds.reset()
        y_true = self.val_ds.classes
        
        # Get predictions
        predictions = self.model.predict(self.val_ds)
        y_pred = np.argmax(predictions, axis=1)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def plot_training_history(self):
        """
        Plot the training history.
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object containing the plots
        """
        if self.history is None:
            print("Model needs to be trained first.")
            return None
            
        print("Plotting training history...")
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
        return fig
    
    def plot_confusion_matrix(self, metrics=None):
        """
        Plot the confusion matrix.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary containing evaluation metrics (optional)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object containing the confusion matrix plot
        """
        if metrics is None:
            metrics = self.evaluate_model()
            
        if metrics is None:
            return None
            
        print("Plotting confusion matrix...")
        
        # Get confusion matrix
        conf_matrix = metrics['confusion_matrix']
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        return plt.gcf()
    
    def generate_performance_report(self, metrics=None):
        """
        Generate a comprehensive performance report.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary containing evaluation metrics (optional)
        
        Returns:
        --------
        report_df : pd.DataFrame
            DataFrame containing performance metrics for each class
        """
        if metrics is None:
            metrics = self.evaluate_model()
            
        if metrics is None:
            return None
        
        print("Generating performance report...")
        
        # Extract class metrics from the classification report
        report = metrics['classification_report']
        
        # Create a dataframe for the report
        data = []
        for class_name in self.class_names:
            class_metrics = report[class_name]
            data.append({
                'Class': class_name,
                'Precision': class_metrics['precision'],
                'Recall': class_metrics['recall'],
                'F1-Score': class_metrics['f1-score'],
                'Support': class_metrics['support']
            })
        
        # Add average metrics
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report:
                data.append({
                    'Class': avg_type,
                    'Precision': report[avg_type]['precision'],
                    'Recall': report[avg_type]['recall'],
                    'F1-Score': report[avg_type]['f1-score'],
                    'Support': report[avg_type]['support']
                })
        
        # Create a DataFrame and save to CSV
        report_df = pd.DataFrame(data)
        report_df.to_csv('classification_metrics.csv', index=False)
        
        # Create a bar chart for class-wise F1 scores
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Class', y='F1-Score', data=report_df[report_df['Class'].isin(self.class_names)])
        plt.title('F1-Score by Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('f1_scores.png')
        plt.show()
        
        return report_df
    
    def export_model(self, format='h5', export_dir='exported_model'):
        """
        Export the trained model in the specified format.
        
        Parameters:
        -----------
        format : str
            Format to export the model in ('h5', 'savedmodel', or 'tflite')
        export_dir : str
            Directory to save the exported model
            
        Returns:
        --------
        export_path : str
            Path to the exported model
        """
        if self.model is None:
            print("Model needs to be trained first.")
            return None
        
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        if format.lower() == 'h5':
            # Save as H5 file
            export_path = os.path.join(export_dir, 'train_classifier.h5')
            self.model.save(export_path)
            print(f"Model exported as H5 file: {export_path}")
            
        elif format.lower() == 'savedmodel':
            # Save as SavedModel
            export_path = os.path.join(export_dir, 'savedmodel')
            self.model.save(export_path)
            print(f"Model exported as SavedModel: {export_path}")
            
            # Save class names
            with open(os.path.join(export_dir, 'class_names.txt'), 'w') as f:
                for class_name in self.class_names:
                    f.write(f"{class_name}\n")
            
        elif format.lower() == 'tflite':
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
            
            # Save the TFLite model
            export_path = os.path.join(export_dir, 'train_classifier.tflite')
            with open(export_path, 'wb') as f:
                f.write(tflite_model)
            print(f"Model exported as TFLite: {export_path}")
            
            # Save class names
            with open(os.path.join(export_dir, 'class_names.txt'), 'w') as f:
                for class_name in self.class_names:
                    f.write(f"{class_name}\n")
                    
        else:
            print(f"Unsupported export format: {format}")
            return None
            
        return export_path
        
    def save_inference_example(self):
        """
        Create and save a Python script for inference with the exported model.
        
        Returns:
        --------
        None
        """
        # Create inference script
        inference_code = """
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def load_class_names(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def predict_image(model_path, image_path, class_names_path, img_height=224, img_width=224):
    # Load the model
    model = load_model(model_path)
    
    # Load class names
    class_names = load_class_names(class_names_path)
    
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    # Display the image and prediction
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class} ({confidence:.2f})')
    plt.axis('off')
    plt.show()
    
    # Return all predictions
    result = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    return result, predicted_class, confidence

# Example usage
if __name__ == "__main__":
    # Replace with your actual paths
    model_path = "exported_model/train_classifier.h5"  # or "exported_model/savedmodel"
    image_path = "path/to/your/test_image.jpg"
    class_names_path = "exported_model/class_names.txt"
    
    # Make prediction
    predictions, predicted_class, confidence = predict_image(
        model_path, image_path, class_names_path)
    
    print(f"Predicted class: {predicted_class} with confidence: {confidence:.2f}")
    print("\\nAll predictions:")
    for class_name, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
        print(f"{class_name}: {prob:.4f}")
"""

        # Save the inference script
        with open('inference_example.py', 'w') as f:
            f.write(inference_code.strip())
            
        print("Inference example script saved as 'inference_example.py'")

# Example usage
def main():
    # Define the data directory
    data_dir = "path/to/train_images_dataset"  # Replace with your dataset path
    
    # Create model instance
    model = TrainClassificationModel(data_dir, img_height=224, img_width=224, batch_size=32)
    
    # Prepare data
    model.prepare_data(validation_split=0.2)
    
    # Build and train model
    model.build_model()
    model.train_model(epochs=50)
    
    # Evaluate model
    metrics = model.evaluate_model()
    
    # Generate visualizations and reports
    model.plot_training_history()
    model.plot_confusion_matrix(metrics)
    model.generate_performance_report(metrics)
    
    # Export model
    model.export_model(format='savedmodel')
    model.save_inference_example()
    
    print("Model training, evaluation, and export process complete.")

if __name__ == "__main__":
    main()
