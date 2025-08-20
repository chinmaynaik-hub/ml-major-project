"""
AI Yoga Trainer - Small Dataset Pose Classifier

This module demonstrates how to train effective pose classification 
with just 100 images per pose using MediaPipe features.

KEY INSIGHT: We're not doing raw image classification!
We're classifying 44-dimensional feature vectors extracted from poses.
This requires much less data than CNN image classification.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import cv2
import os
from typing import Dict, List, Tuple
from pose_detector import PoseDetector
import json

class SmallDatasetPoseClassifier:
    """
    Pose classifier optimized for small datasets (100 images per pose).
    
    STRATEGY:
    1. Use MediaPipe feature extraction (not raw images)
    2. Apply data augmentation
    3. Use ensemble methods (Random Forest)
    4. Cross-validation for robust evaluation
    """
    
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.scaler = StandardScaler()
        self.classifier = None
        self.pose_names = []
        
        # Sanskrit to English mapping
        self.asana_mapping = {
            'tadasana': 'ताड़ासन',
            'vrikshasana': 'वृक्षासन', 
            'trikonasana': 'त्रिकोणासन',
            'bhujangasana': 'भुजंगासन',
            'balasana': 'बालासन'
        }
    
    def prepare_dataset_from_images(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pose features from image dataset.
        
        Expected structure:
        dataset_path/
        ├── tadasana/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        ├── vrikshasana/
        │   ├── img1.jpg
        │   └── ...
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            X: Feature vectors (n_samples, 44)
            y: Labels (n_samples,)
        """
        X = []
        y = []
        
        print("Extracting pose features from images...")
        
        # Get all pose directories
        pose_dirs = [d for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d))]
        
        self.pose_names = sorted(pose_dirs)
        print(f"Found poses: {self.pose_names}")
        
        for pose_idx, pose_name in enumerate(self.pose_names):
            pose_path = os.path.join(dataset_path, pose_name)
            image_files = [f for f in os.listdir(pose_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Processing {pose_name}: {len(image_files)} images")
            
            successful_extractions = 0
            
            for img_file in image_files:
                img_path = os.path.join(pose_path, img_file)
                
                try:
                    # Load and process image
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    # Extract pose features
                    pose_data = self.pose_detector.detect_pose(image)
                    
                    if pose_data['pose_detected'] and pose_data['confidence'] > 0.5:
                        features = self.pose_detector.get_pose_features(
                            pose_data['keypoints'], 
                            pose_data['angles']
                        )
                        
                        X.append(features)
                        y.append(pose_idx)
                        successful_extractions += 1
                        
                        # Apply data augmentation for small datasets
                        augmented_features = self._augment_features(features)
                        for aug_features in augmented_features:
                            X.append(aug_features)
                            y.append(pose_idx)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            print(f"Successfully extracted {successful_extractions} poses from {pose_name}")
            print(f"Total samples (with augmentation): {successful_extractions * 4}")
        
        return np.array(X), np.array(y)
    
    def _augment_features(self, features: np.ndarray) -> List[np.ndarray]:
        """
        Apply data augmentation to pose features.
        
        AUGMENTATION STRATEGIES:
        1. Small random noise (simulate detection variations)
        2. Slight coordinate shifts (simulate camera position changes)
        3. Angle variations (simulate pose micro-variations)
        
        Args:
            features: Original feature vector
            
        Returns:
            List of augmented feature vectors
        """
        augmented = []
        
        # Strategy 1: Add small random noise
        noise_levels = [0.01, 0.02, 0.015]
        for noise_level in noise_levels:
            noise = np.random.normal(0, noise_level, features.shape)
            augmented_features = features + noise
            augmented_features = np.clip(augmented_features, 0, 1)  # Keep in valid range
            augmented.append(augmented_features)
        
        return augmented
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train pose classifier with cross-validation.
        
        ALGORITHM CHOICE:
        - Random Forest: Works well with small datasets, handles overfitting
        - Alternative: SVM with RBF kernel for non-linear boundaries
        
        Args:
            X: Feature vectors
            y: Labels
            
        Returns:
            Training results and metrics
        """
        print(f"Training with {len(X)} samples, {len(np.unique(y))} poses")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try multiple algorithms
        algorithms = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'  # Handle potential class imbalance
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=min(5, len(X_train) // len(self.pose_names))  # Adaptive K
            )
        }
        
        results = {}
        best_score = 0
        best_algorithm = None
        
        print("\nComparing algorithms:")
        for name, clf in algorithms.items():
            # Cross-validation
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
            mean_cv_score = np.mean(cv_scores)
            
            print(f"{name}: CV Score = {mean_cv_score:.3f} ± {np.std(cv_scores):.3f}")
            
            results[name] = {
                'cv_score': mean_cv_score,
                'cv_std': np.std(cv_scores),
                'classifier': clf
            }
            
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_algorithm = name
        
        # Train best algorithm on full training data
        print(f"\nBest algorithm: {best_algorithm}")
        self.classifier = results[best_algorithm]['classifier']
        self.classifier.fit(X_train_scaled, y_train)
        
        # Final evaluation
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        # Detailed test results
        y_pred = self.classifier.predict(X_test_scaled)
        classification_rep = classification_report(y_test, y_pred, 
                                                 target_names=self.pose_names)
        
        training_results = {
            'best_algorithm': best_algorithm,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_results': results,
            'classification_report': classification_rep,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_poses': len(self.pose_names)
        }
        
        print(f"\nFinal Results:")
        print(f"Training Accuracy: {train_score:.3f}")
        print(f"Testing Accuracy: {test_score:.3f}")
        print(f"\nClassification Report:")
        print(classification_rep)
        
        return training_results
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Predict pose from image.
        
        Args:
            image: Input BGR image
            
        Returns:
            Prediction results with confidence and Sanskrit name
        """
        if self.classifier is None:
            raise ValueError("Model not trained yet!")
        
        # Extract pose features
        pose_data = self.pose_detector.detect_pose(image)
        
        if not pose_data['pose_detected']:
            return {
                'pose_detected': False,
                'prediction': None,
                'confidence': 0.0,
                'sanskrit_name': None
            }
        
        # Get features and predict
        features = self.pose_detector.get_pose_features(
            pose_data['keypoints'], 
            pose_data['angles']
        )
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prediction = self.classifier.predict(features_scaled)[0]
        confidence = np.max(self.classifier.predict_proba(features_scaled))
        
        pose_name = self.pose_names[prediction]
        sanskrit_name = self.asana_mapping.get(pose_name, pose_name)
        
        return {
            'pose_detected': True,
            'prediction': pose_name,
            'confidence': confidence,
            'sanskrit_name': sanskrit_name,
            'processed_image': pose_data['processed_image']
        }
    
    def save_model(self, filepath: str):
        """Save trained model and preprocessing components."""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'pose_names': self.pose_names,
            'asana_mapping': self.asana_mapping
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and preprocessing components."""
        model_data = joblib.load(filepath)
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.pose_names = model_data['pose_names']
        self.asana_mapping = model_data['asana_mapping']
        print(f"Model loaded from {filepath}")


# PRACTICAL EXAMPLE USAGE
def train_small_dataset_example():
    """
    Example of training with small dataset (100 images per pose).
    
    DATASET REQUIREMENTS:
    - 5 poses × 100 images = 500 base images
    - With augmentation: ~2000 training samples
    - Expected accuracy: 85-95% with good quality images
    """
    
    # Initialize classifier
    classifier = SmallDatasetPoseClassifier()
    
    # Prepare dataset (replace with your dataset path)
    dataset_path = "./yoga_dataset_small"
    
    try:
        X, y = classifier.prepare_dataset_from_images(dataset_path)
        print(f"Dataset prepared: {len(X)} samples, {len(np.unique(y))} classes")
        
        # Train model
        results = classifier.train(X, y)
        
        # Save model
        classifier.save_model("./models/small_dataset_yoga_classifier.pkl")
        
        return classifier, results
        
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}")
        print("Please organize your dataset as described in the docstring")
        return None, None


if __name__ == "__main__":
    # Run training example
    classifier, results = train_small_dataset_example()
    
    if classifier and results:
        print("\nTraining completed successfully!")
        print(f"Test Accuracy: {results['test_accuracy']:.3f}")
        print("Model ready for real-time pose prediction!")
