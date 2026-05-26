"""
Training script for pneumonia detection model
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
from data_loader import PneumoniaDataLoader
from model import PneumoniaCNNModel, compile_model


class ModelTrainer:
    def __init__(self, dataset_path, model_save_path='models/'):
        """
        Initialize trainer
        
        Args:
            dataset_path: Path to dataset
            model_save_path: Path to save models
        """
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs('logs/', exist_ok=True)
        
    def train_model(self, model, X_train, X_val, y_train, y_val, 
                   epochs=50, batch_size=32, model_name='pneumonia_cnn'):
        """
        Train the model
        
        Args:
            model: Compiled Keras model
            X_train: Training images
            X_val: Validation images
            y_train: Training labels
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            model_name: Name for saving model
            
        Returns:
            Training history
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(self.model_save_path, f'{model_name}_{timestamp}.h5')
        csv_log_path = os.path.join('logs/', f'{model_name}_{timestamp}.csv')
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=f'logs/tensorboard_{timestamp}',
                histogram_freq=1,
                write_graph=True
            ),
            
            # CSV logging
            CSVLogger(csv_log_path, append=True)
        ]
        
        print(f"\nTraining model: {model_name}")
        print(f"Model will be saved to: {checkpoint_path}")
        print(f"Logs saved to: {csv_log_path}\n")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            class_weight={0: 1, 1: 1.2}  # Handle class imbalance
        )
        
        return history, checkpoint_path
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """
        Plot training history
        
        Args:
            history: Training history from model.fit()
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to: {save_path}")
        
        plt.show()


def main():
    """Main training script"""
    
    # Configuration
    DATASET_PATH = 'data/'  # Update with your dataset path
    IMG_SIZE = (224, 224)
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Load data
    print("Loading dataset...")
    loader = PneumoniaDataLoader(DATASET_PATH, img_size=IMG_SIZE)
    images, labels = loader.load_images_from_folder()
    
    # Split data
    print("\nSplitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
        test_size=0.2, val_size=0.2
    )
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Build and compile model
    print("\nBuilding model...")
    model = PneumoniaCNNModel.build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
    model = compile_model(model)
    model.summary()
    
    # Train model
    print("\nStarting training...")
    trainer = ModelTrainer(DATASET_PATH)
    history, model_path = trainer.train_model(
        model, X_train, X_val, y_train, y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, model_name='pneumonia_cnn'
    )
    
    # Plot results
    print("\nPlotting training history...")
    trainer.plot_training_history(history, save_path='plots/training_history.png')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")


if __name__ == '__main__':
    main()
