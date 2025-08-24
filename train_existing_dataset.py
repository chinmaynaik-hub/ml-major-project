#!/usr/bin/env python3
"""
Train Yoga AI model with existing dataset
Adapted for the current dataset structure with comprehensive pose collection
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
import logging
from typing import List, Dict, Tuple
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from yoga_ai_trainer.backend.models.pose_classifier_small import YogaPoseClassifier
except ImportError as e:
    logger.error(f"Failed to import YogaPoseClassifier: {e}")
    sys.exit(1)

class ExistingDatasetTrainer:
    """Train the yoga pose classifier using the existing dataset structure."""
    
    def __init__(self, dataset_root="yoga_ai_trainer/backend/data/raw/yoga_poses/dataset"):
        self.dataset_root = Path(dataset_root)
        self.min_images_per_pose = 30  # Minimum images needed for training
        
    def analyze_dataset(self) -> Dict:
        """Analyze the existing dataset to find suitable poses for training."""
        logger.info("Analyzing existing dataset...")
        
        pose_stats = {}
        suitable_poses = []
        
        for pose_dir in self.dataset_root.iterdir():
            if pose_dir.is_dir() and pose_dir.name not in ['train', 'val', 'test']:
                # Count images
                image_files = list(pose_dir.glob("*.jpg")) + list(pose_dir.glob("*.jpeg")) + list(pose_dir.glob("*.png"))
                count = len(image_files)
                
                pose_stats[pose_dir.name] = count
                
                if count >= self.min_images_per_pose:
                    suitable_poses.append((pose_dir.name, count))
        
        # Sort by image count (descending)
        suitable_poses.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'all_poses': pose_stats,
            'suitable_poses': suitable_poses,
            'total_poses': len(pose_stats),
            'trainable_poses': len(suitable_poses)
        }
    
    def select_training_poses(self, analysis: Dict, max_poses: int = 10) -> List[str]:
        """Select the best poses for training."""
        suitable_poses = analysis['suitable_poses']
        
        # Take top poses with most images
        selected = suitable_poses[:max_poses]
        pose_names = [pose[0] for pose in selected]
        
        logger.info(f"Selected {len(pose_names)} poses for training:")
        for pose, count in selected:
            logger.info(f"  - {pose}: {count} images")
        
        return pose_names
    
    def prepare_training_data(self, selected_poses: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data from selected poses."""
        logger.info("Preparing training data...")
        
        classifier = YogaPoseClassifier()
        
        X = []
        y = []
        
        # Create pose name mapping
        pose_names = selected_poses
        pose_to_idx = {pose: idx for idx, pose in enumerate(pose_names)}
        
        logger.info(f"Processing {len(selected_poses)} pose classes...")
        
        for pose_name in selected_poses:
            pose_dir = self.dataset_root / pose_name
            pose_idx = pose_to_idx[pose_name]
            
            # Get all image files
            image_files = list(pose_dir.glob("*.jpg")) + list(pose_dir.glob("*.jpeg")) + list(pose_dir.glob("*.png"))
            
            logger.info(f"Processing {pose_name}: {len(image_files)} images")
            successful_extractions = 0
            
            for img_file in image_files:
                try:
                    import cv2
                    
                    # Load image
                    image = cv2.imread(str(img_file))
                    if image is None:
                        continue
                    
                    # Extract pose features
                    pose_data = classifier.pose_detector.detect_pose(image)
                    
                    if pose_data['pose_detected'] and pose_data['confidence'] > 0.5:
                        features = classifier.pose_detector.get_pose_features(
                            pose_data['keypoints'], 
                            pose_data['angles']
                        )
                        
                        X.append(features)
                        y.append(pose_idx)
                        successful_extractions += 1
                        
                        # Apply data augmentation for better training
                        augmented_features = classifier._augment_features(features)
                        for aug_features in augmented_features:
                            X.append(aug_features)
                            y.append(pose_idx)
                
                except Exception as e:
                    logger.warning(f"Error processing {img_file}: {e}")
                    continue
            
            logger.info(f"✅ {pose_name}: {successful_extractions} successful extractions ({successful_extractions * 4} total with augmentation)")
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Final dataset: {len(X)} samples, {len(pose_names)} classes")
        
        return X, y, pose_names

def main():
    """Main training function."""
    print("🧘‍♀️ Yoga AI Trainer - Training with Existing Dataset")
    print("=" * 60)
    
    trainer = ExistingDatasetTrainer()
    
    # Check if dataset exists
    if not trainer.dataset_root.exists():
        print(f"❌ Dataset directory not found: {trainer.dataset_root}")
        print("Please ensure your dataset is in the correct location.")
        return
    
    # Analyze dataset
    print("🔍 Analyzing dataset...")
    analysis = trainer.analyze_dataset()
    
    print(f"\n📊 Dataset Analysis Results:")
    print(f"   Total pose directories: {analysis['total_poses']}")
    print(f"   Poses with sufficient data (≥{trainer.min_images_per_pose} images): {analysis['trainable_poses']}")
    
    if analysis['trainable_poses'] == 0:
        print("❌ No poses have sufficient data for training.")
        print(f"   Need at least {trainer.min_images_per_pose} images per pose.")
        return
    
    print(f"\n📝 Top poses by image count:")
    for i, (pose, count) in enumerate(analysis['suitable_poses'][:15]):
        print(f"   {i+1:2d}. {pose}: {count} images")
    
    # Select poses for training
    selected_poses = trainer.select_training_poses(analysis, max_poses=12)
    
    if not selected_poses:
        print("❌ No suitable poses found for training.")
        return
    
    print(f"\n🎯 Selected poses for training: {len(selected_poses)}")
    
    try:
        # Prepare training data
        print("\n⚙️ Preparing training data...")
        X, y, pose_names = trainer.prepare_training_data(selected_poses)
        
        if len(X) == 0:
            print("❌ No training data could be extracted.")
            print("   This might be due to:")
            print("   - Images don't contain detectable human poses")
            print("   - Poor image quality")
            print("   - Incompatible image formats")
            return
        
        # Initialize classifier and train
        print(f"\n🚀 Starting training with {len(X)} samples...")
        classifier = YogaPoseClassifier()
        classifier.pose_names = pose_names  # Update pose names
        
        # Train the model
        results = classifier.train(X, y)
        
        # Save the trained model
        model_path = f"yoga_model_trained_{int(time.time())}.pkl"
        classifier.save_model(model_path)
        
        # Save pose mapping
        pose_mapping = {
            'pose_names': pose_names,
            'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_stats': analysis,
            'training_results': {
                'accuracy': float(results.get('test_accuracy', 0)),
                'algorithm': results.get('best_algorithm', 'unknown'),
                'samples': len(X),
                'classes': len(pose_names)
            }
        }
        
        mapping_path = f"pose_mapping_{int(time.time())}.json"
        with open(mapping_path, 'w') as f:
            json.dump(pose_mapping, f, indent=2)
        
        # Print results
        print("\n🎉 Training completed successfully!")
        print("=" * 60)
        print(f"🎯 Test Accuracy: {results.get('test_accuracy', 0):.3f}")
        print(f"🤖 Best Algorithm: {results.get('best_algorithm', 'unknown')}")
        print(f"📊 Training Samples: {len(X)}")
        print(f"📝 Pose Classes: {len(pose_names)}")
        print(f"💾 Model saved as: {model_path}")
        print(f"🗂️ Pose mapping saved as: {mapping_path}")
        
        if results.get('test_accuracy', 0) >= 0.80:
            print("✅ Model achieves good accuracy (≥80%)")
        elif results.get('test_accuracy', 0) >= 0.70:
            print("⚠️ Model has moderate accuracy (70-80%)")
            print("   Consider collecting more training data for better performance")
        else:
            print("⚠️ Model has low accuracy (<70%)")
            print("   Recommendations:")
            print("   - Collect more high-quality images")
            print("   - Ensure images show clear pose positions")
            print("   - Remove low-quality or ambiguous images")
        
        print(f"\n📝 Trained pose classes:")
        for i, pose in enumerate(pose_names):
            print(f"   {i+1:2d}. {pose}")
        
        # Database information
        print(f"\n💾 Database Information:")
        print(f"   This project uses the filesystem for data storage:")
        print(f"   - Dataset: {trainer.dataset_root}")
        print(f"   - Models: {model_path}")
        print(f"   - No external database required")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n❌ Training failed: {e}")
        print("Check the logs above for detailed error information.")

if __name__ == "__main__":
    main()
