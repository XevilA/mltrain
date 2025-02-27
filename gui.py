import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QFileDialog, QVBoxLayout, QWidget, QTextEdit,
                            QProgressBar, QHBoxLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

class PredictionThread(QThread):
    prediction_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, model_path, image_path):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path

    def run(self):
        try:
            # Load model and class names
            self.model = load_model(self.model_path)
            self.class_names = ["tail pass", "tail fail", "flake pass", "flake fail"]
            
            # Load and preprocess image
            img = image.load_img(self.image_path, target_size=(384, 384))
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img_array)
            confidence = np.max(predictions)
            class_index = np.argmax(predictions)
            
            result = {
                'class': self.class_names[class_index],
                'confidence': float(confidence),
                'image_path': self.image_path
            }
            self.prediction_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))

class PlasticClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.initUI()
        self.load_model()
        
    def initUI(self):
        self.setWindowTitle('Plastic Defect Classifier')
        self.setGeometry(100, 100, 800, 600)
        
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Image Display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        layout.addWidget(self.image_label)
        
        # Progress Bar
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)
        
        # Results Display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.btn_load = QPushButton('Load Image')
        self.btn_load.clicked.connect(self.load_image)
        button_layout.addWidget(self.btn_load)
        
        self.btn_classify = QPushButton('Classify')
        self.btn_classify.clicked.connect(self.classify_image)
        self.btn_classify.setEnabled(False)
        button_layout.addWidget(self.btn_classify)
        
        layout.addLayout(button_layout)
        
        # Status Bar
        self.statusBar().showMessage('Ready')
        
    def load_model(self):
        try:
            self.model = load_model('plastic_classifier.h5')
            self.statusBar().showMessage('Model loaded successfully')
        except Exception as e:
            self.statusBar().showMessage(f'Error loading model: {str(e)}')
            self.btn_classify.setEnabled(False)
        
    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp)", 
            options=options
        )
        
        if file_path:
            self.current_image = file_path
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.btn_classify.setEnabled(True)
            self.results_text.clear()
        
    def classify_image(self):
        if not hasattr(self, 'current_image'):
            return
            
        self.thread = PredictionThread('plastic_classifier.h5', self.current_image)
        self.thread.prediction_signal.connect(self.show_results)
        self.thread.error_signal.connect(self.show_error)
        self.thread.start()
        
        self.progress.show()
        self.btn_classify.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.statusBar().showMessage('Classifying...')
        
    def show_results(self, result):
        self.progress.hide()
        self.btn_classify.setEnabled(True)
        self.btn_load.setEnabled(True)
        
        text = f"""Classification Result:
        - Class: {result['class']}
        - Confidence: {result['confidence']:.2%}
        - Image Path: {result['image_path']}
        """
        self.results_text.setPlainText(text)
        self.statusBar().showMessage('Classification complete')
        
    def show_error(self, error_msg):
        self.progress.hide()
        self.btn_classify.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.results_text.setPlainText(f"Error: {error_msg}")
        self.statusBar().showMessage('Error occurred')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PlasticClassifierApp()
    window.show()
    sys.exit(app.exec_())
