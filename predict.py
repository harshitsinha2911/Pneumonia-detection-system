"""
Prediction script for making predictions on new X-ray images
"""
import os
import numpy as np
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt


class PneumoniaPredictor:
    def __init__(self, model_path, img_size=(224, 224)):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model
            img_size: Image size
        """
        self.model = keras.models.load_model(model_path)
        self.img_size = img_size
        
    def load_image(self, image_path):
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image
            
        Returns:
            Preprocessed image array
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, image_path, threshold=0.5):
        """
        Make prediction for single image
        
        Args:
            image_path: Path to image
            threshold: Classification threshold
            
        Returns:
            Prediction (0=Normal, 1=Pneumonia) and confidence
        """
        img = self.load_image(image_path)
        prediction = self.model.predict(img, verbose=0)[0][0]
        
        label = 'Pneumonia' if prediction >= threshold else 'Normal'
        confidence = prediction if prediction >= threshold else 1 - prediction
        
        return label, confidence, prediction
    
    def predict_batch(self, image_paths, threshold=0.5):
        """
        Make predictions for multiple images
        
        Args:
            image_paths: List of image paths
            threshold: Classification threshold
            
        Returns:
            List of (label, confidence) tuples
        """
        results = []
        for image_path in image_paths:
            label, confidence, _ = self.predict(image_path, threshold)
            results.append({
                'image': image_path,
                'label': label,
                'confidence': confidence
            })
        return results
    
    def visualize_prediction(self, image_path, threshold=0.5, save_path=None):
        """
        Visualize prediction with image
        
        Args:
            image_path: Path to image
            threshold: Classification threshold
            save_path: Path to save visualization
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label, confidence, prob = self.predict(image_path, threshold)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap='gray')
        plt.title(f'Prediction: {label} (Confidence: {confidence:.2%})', 
                 fontsize=14, fontweight='bold',
                 color='green' if label == 'Normal' else 'red')
        plt.axis('off')
        
        # Add probability info
        normal_prob = 1 - prob
        pneumonia_prob = prob
        info_text = f'Normal: {normal_prob:.2%}\nPneumonia: {pneumonia_prob:.2%}'
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def predict_from_folder(self, folder_path, threshold=0.5):
        """
        Make predictions for all images in a folder
        
        Args:
            folder_path: Path to folder with images
            threshold: Classification threshold
            
        Returns:
            List of results
        """
        results = []
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                label, confidence, _ = self.predict(image_path, threshold)
                results.append({
                    'image': filename,
                    'path': image_path,
                    'label': label,
                    'confidence': confidence
                })
        
        return results


def main():
    """Example usage"""
    
    # Load model
    model_path = 'models/pneumonia_cnn_final.h5'  # Update with your model path
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please train a model first using train.py")
        return
    
    predictor = PneumoniaPredictor(model_path)
    
    # Example 1: Predict single image
    print("Example 1: Single image prediction")
    print("-" * 50)
    test_image = 'data/NORMAL/image1.jpg'  # Update with actual image path
    
    if os.path.exists(test_image):
        label, confidence, _ = predictor.predict(test_image)
        print(f"Image: {test_image}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")
        predictor.visualize_prediction(test_image)
    else:
        print(f"Test image not found: {test_image}")
    
    # Example 2: Predict folder
    print("\n\nExample 2: Batch prediction from folder")
    print("-" * 50)
    test_folder = 'data/NORMAL'
    
    if os.path.exists(test_folder):
        results = predictor.predict_from_folder(test_folder)
        print(f"Predictions for images in {test_folder}:")
        for result in results[:5]:  # Show first 5
            print(f"  {result['image']}: {result['label']} ({result['confidence']:.2%})")
        
        # Summary
        normal_count = sum(1 for r in results if r['label'] == 'Normal')
        pneumonia_count = sum(1 for r in results if r['label'] == 'Pneumonia')
        print(f"\nSummary: {normal_count} Normal, {pneumonia_count} Pneumonia (out of {len(results)})")
    else:
        print(f"Folder not found: {test_folder}")


if __name__ == '__main__':
    main()
