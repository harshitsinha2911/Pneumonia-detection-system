# Pneumonia Detection using CNN

A comprehensive deep learning project for detecting pneumonia from chest X-ray images using Convolutional Neural Networks.

## Overview

This project implements a complete pipeline for pneumonia detection from chest X-ray images using:
- **Custom CNN Architecture**: A custom-built convolutional neural network
- **Transfer Learning**: Pre-trained models (ResNet50, VGG16, DenseNet121)
- **Advanced Evaluation**: Confusion matrix, ROC curves, precision-recall curves, etc.

## Project Structure

```
pneumonia detection CNN/
├── data/                          # Dataset folder (create and populate this)
│   ├── NORMAL/                   # Normal X-ray images
│   └── PNEUMONIA/                # Pneumonia X-ray images
├── models/                        # Saved model checkpoints
├── logs/                          # Training logs
├── plots/                         # Generated plots and visualizations
├── Pneumonia_Detection_CNN.ipynb # Main Jupyter notebook
├── requirements.txt              # Python dependencies
├── data_loader.py               # Data loading and preprocessing
├── model.py                     # Model architectures
├── train.py                     # Training script for custom CNN
├── train_transfer_learning.py   # Transfer learning training
├── predict.py                   # Prediction on new images
├── evaluator.py                 # Model evaluation utilities
└── README.md                    # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- GPU (optional, but recommended for faster training)

### Setup

1. **Create the project directory structure:**
```bash
mkdir -p pneumonia\ detection\ CNN
cd pneumonia\ detection\ CNN
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install tensorflow==2.14.0 keras==2.14.0 numpy==1.24.3 pandas==2.0.3
pip install scikit-learn==1.3.0 matplotlib==3.7.2 seaborn==0.12.2
pip install opencv-python==4.8.0.74 Pillow==10.0.0
```

## Dataset Preparation

1. **Organize your dataset** in the following structure:
```
data/
├── NORMAL/
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
└── PNEUMONIA/
    ├── image1.jpeg
    ├── image2.jpeg
    └── ...
```

2. **Dataset requirements:**
   - Images should be chest X-ray images (JPEG, PNG, or JPG format)
   - Images will be automatically resized to 224x224 pixels
   - Both grayscale and color images are supported

3. **Dataset sources:**
   - [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
   - [Mendeley Data - Chest X-ray Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)

## Usage

### Option 1: Using Jupyter Notebook (Recommended)

1. **Start Jupyter Notebook:**
```bash
jupyter notebook
```

2. **Open `Pneumonia_Detection_CNN.ipynb`**

3. **Follow the step-by-step cells:**
   - Data loading and exploration
   - Model building
   - Training
   - Evaluation and visualization

### Option 2: Command Line Scripts

#### Train Custom CNN Model:
```bash
python train.py
```

This script will:
- Load the dataset from `data/` folder
- Split data into train/validation/test sets
- Build and train a custom CNN model
- Save the best model checkpoint
- Evaluate on test set
- Generate training plots

#### Train Transfer Learning Models:
```bash
python train_transfer_learning.py
```

This script trains three pre-trained models:
- ResNet50
- VGG16
- DenseNet121

#### Make Predictions:
```bash
python predict.py
```

This script demonstrates how to:
- Load a trained model
- Make predictions on single images
- Batch predict from a folder
- Visualize predictions

## Model Architectures

### Custom CNN Model
- 4 convolutional blocks with 32, 64, 128, and 256 filters
- Batch normalization and dropout for regularization
- Global average pooling
- 3 dense layers with 512, 256, and 1 neurons
- Binary classification output with sigmoid activation

### Transfer Learning Models
- **ResNet50**: Deep residual network
- **VGG16**: 16-layer visual geometry group network
- **DenseNet121**: Densely connected convolutional network

## Training Configuration

Default parameters (can be modified in scripts):
- **Image size**: 224x224 pixels
- **Batch size**: 32
- **Epochs**: 50
- **Learning rate**: 1e-3 (custom CNN), 1e-4 (transfer learning)
- **Optimizer**: Adam
- **Loss function**: Binary Crossentropy
- **Metrics**: Accuracy, Precision, Recall, AUC

## Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **Specificity**: True negatives / (True negatives + False positives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC (Area Under Curve)**: ROC curve area
- **Confusion Matrix**: Visualization of classification results

## Visualizations Generated

The project generates the following plots:
- `training_history.png`: Training/validation accuracy, loss, precision, recall
- `confusion_matrix.png`: Confusion matrix heatmap
- `roc_curve.png`: Receiver Operating Characteristic curve
- `precision_recall_curve.png`: Precision-Recall trade-off curve
- `prediction_distribution.png`: Distribution of prediction probabilities
- `sample_predictions.png`: Sample images with predictions

## Saved Artifacts

After training:
- **Models**: Saved in `models/` folder as `.h5` files
- **Logs**: Training logs saved in `logs/` folder as `.csv` files
- **Plots**: All visualizations saved in `plots/` folder
- **Checkpoints**: Best model checkpoint based on validation AUC

## How to Use Trained Model

```python
from predict import PneumoniaPredictor

# Load predictor
predictor = PneumoniaPredictor('models/pneumonia_cnn_final.h5')

# Predict single image
label, confidence, prob = predictor.predict('path/to/xray.jpg')
print(f"Prediction: {label}, Confidence: {confidence:.2%}")

# Visualize
predictor.visualize_prediction('path/to/xray.jpg', save_path='prediction.png')

# Batch predict
results = predictor.predict_from_folder('path/to/folder')
for result in results:
    print(f"{result['image']}: {result['label']} ({result['confidence']:.2%})")
```

## Performance Tips

1. **GPU Acceleration**: Install CUDA and cuDNN for faster training
```bash
pip install tensorflow[and-cuda]
```

2. **Data Augmentation**: More aggressive augmentation is applied during training

3. **Class Imbalance Handling**: Class weights are used to handle imbalanced datasets

4. **Early Stopping**: Training stops if validation loss doesn't improve for 10 epochs

5. **Learning Rate Reduction**: Learning rate is reduced by 50% if validation loss plateaus

## Troubleshooting

### Out of Memory (OOM) Error
- Reduce batch size in training scripts (e.g., from 32 to 16)
- Reduce image size (e.g., from 224x224 to 192x192)
- Use gradient accumulation

### Model not converging
- Increase learning rate
- Ensure data is properly normalized
- Check for class imbalance
- Try different architectures (transfer learning)

### Dataset not found
- Ensure `data/` folder exists in project directory
- Check folder structure: `data/NORMAL/` and `data/PNEUMONIA/`
- Verify image file formats (.jpg, .jpeg, .png)

## Advanced Usage

### Custom Model Architecture
Edit `model.py` to modify the CNN architecture:
```python
# Add more layers, change filter sizes, etc.
```

### Hyperparameter Tuning
Modify parameters in training scripts:
- Learning rate
- Batch size
- Number of epochs
- Dropout rates

### Model Deployment
Save model in different formats:
```python
# H5 format (already done)
model.save('model.h5')

# TensorFlow SavedModel
model.save('model_savedmodel')

# ONNX format
import tf2onnx
onnx_model = tf2onnx.convert.from_keras(model)
```

## References

1. He et al. (2015) - ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. Simonyan & Zisserman (2014) - VGG: [Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556)
3. Huang et al. (2016) - DenseNet: [Densely Connected Networks](https://arxiv.org/abs/1608.06993)

## Citation

If you use this project in your research, please cite:
```
@project{pneumonia_detection_cnn,
  title={Pneumonia Detection using CNN},
  year={2024}
}
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or support, please open an issue on the project repository.

---

**Note**: This model is for educational and research purposes. For medical diagnosis, always consult with qualified healthcare professionals.
