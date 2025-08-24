#!/usr/bin/env python3
"""
Yoga AI Trainer - Enhanced Model Training Script
Trains the yoga pose classifier on the collected dataset with advanced techniques
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
import time
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from yoga_ai_trainer.backend.models.pose_classifier_small import YogaPoseClassifier
except ImportError:
    logger.error("Failed to import YogaPoseClassifier - ensure you're running from project root")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Yoga AI pose classifier model")
    
    parser.add_argument("--dataset", type=str, 
                        default="yoga_ai_trainer/backend/data/raw/yoga_poses/dataset",
                        help="Path to the dataset directory")
    
    parser.add_argument("--output", type=str,
                        default="yoga_ai_trainer/backend/models/trained_models",
                        help="Directory to save the trained model")
    
    parser.add_argument("--model-name", type=str,
                        default="yoga_classifier_v1",
                        help="Name for the trained model file")
    
    parser.add_argument("--use-splits", action="store_true",
                        help="Use existing train/val/test splits instead of creating new ones")
    
    parser.add_argument("--algorithms", type=str, default="all",
                        choices=["all", "rf", "svm", "knn", "dt", "mlp"],
                        help="ML algorithms to use for training")
    
    parser.add_argument("--hyperopt", action="store_true",
                        help="Use hyperparameter optimization")
    
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable generating plots")
    
    parser.add_argument("--augmentation", action="store_true",
                        help="Apply data augmentation during training")
    
    return parser.parse_args()

def create_results_directory(args):
    """Create directories for results and trained models."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create base output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped run directory
    run_dir = output_dir / f"run_{timestamp}_{args.model_name}"
    run_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    return run_dir, plots_dir

def visualize_training_results(results, classifier, plots_dir):
    """Generate visualizations for training results."""
    logger.info("Generating training result visualizations")
    
    # 1. Confusion Matrix
    y_true = results.get('y_test', [])
    y_pred = results.get('y_pred', [])
    
    if len(y_true) > 0 and len(y_pred) > 0:
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=classifier.pose_names,
            yticklabels=classifier.pose_names
        )
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(plots_dir / "confusion_matrix.png", dpi=300)
        plt.close()
    
    # 2. Feature Importance (if available)
    if hasattr(classifier.classifier, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        feature_importance = classifier.classifier.feature_importances_
        # Sort importances
        indices = np.argsort(feature_importance)[::-1]
        
        # Plot top 30 features
        top_n = min(30, len(feature_importance))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(top_n), feature_importance[indices[:top_n]], align='center')
        plt.xticks(range(top_n), [f"Feature {idx}" for idx in indices[:top_n]], rotation=90)
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance.png", dpi=300)
        plt.close()
    
    # 3. Algorithm Comparison
    if 'cv_results' in results:
        cv_results = results['cv_results']
        algorithms = list(cv_results.keys())
        cv_scores = [cv_results[alg]['cv_score'] for alg in algorithms]
        cv_stds = [cv_results[alg]['cv_std'] for alg in algorithms]
        
        plt.figure(figsize=(10, 6))
        plt.bar(algorithms, cv_scores, yerr=cv_stds, capsize=10, alpha=0.7)
        plt.title('Cross-Validation Scores by Algorithm')
        plt.ylabel('Mean CV Score')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plots_dir / "algorithm_comparison.png", dpi=300)
        plt.close()
    
    # 4. Learning Curve if available
    if 'learning_curve' in results:
        train_sizes = results['learning_curve']['train_sizes']
        train_scores = results['learning_curve']['train_scores']
        val_scores = results['learning_curve']['val_scores']
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color='g', label='Validation score')
        plt.title('Learning Curve')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / "learning_curve.png", dpi=300)
        plt.close()
    
    logger.info(f"Visualizations saved to {plots_dir}")

def save_training_summary(results, classifier, run_dir):
    """Save training summary and metadata."""
    summary = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': classifier.__class__.__name__,
        'algorithm': results.get('best_algorithm', 'Unknown'),
        'metrics': {
            'train_accuracy': float(results.get('train_accuracy', 0)),
            'test_accuracy': float(results.get('test_accuracy', 0)),
            'cv_score': float(results.get('cv_score', 0)),
        },
        'dataset': {
            'n_samples': int(results.get('n_samples', 0)),
            'n_features': int(results.get('n_features', 0)),
            'n_poses': int(results.get('n_poses', 0)),
            'poses': classifier.pose_names
        }
    }
    
    # Save summary as JSON
    with open(run_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed report as text
    with open(run_dir / "training_report.txt", "w") as f:
        f.write(f"YOGA AI TRAINER - MODEL TRAINING REPORT\n")
        f.write(f"=====================================\n\n")
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Model: {summary['model_name']}\n")
        f.write(f"Algorithm: {summary['algorithm']}\n\n")
        
        f.write(f"METRICS\n")
        f.write(f"-------\n")
        f.write(f"Training Accuracy: {summary['metrics']['train_accuracy']:.4f}\n")
        f.write(f"Testing Accuracy:  {summary['metrics']['test_accuracy']:.4f}\n")
        f.write(f"CV Score:          {summary['metrics']['cv_score']:.4f}\n\n")
        
        f.write(f"DATASET\n")
        f.write(f"-------\n")
        f.write(f"Samples:  {summary['dataset']['n_samples']}\n")
        f.write(f"Features: {summary['dataset']['n_features']}\n")
        f.write(f"Poses:    {summary['dataset']['n_poses']}\n\n")
        
        f.write(f"Pose Classes:\n")
        for i, pose in enumerate(summary['dataset']['poses']):
            f.write(f"  {i+1}. {pose}\n")
        
        f.write(f"\nDETAILED CLASSIFICATION REPORT\n")
        f.write(f"------------------------------\n")
        f.write(results.get('classification_report', 'Not available'))
    
    logger.info(f"Training summary saved to {run_dir}")

def train_model(args):
    """Train the yoga pose classifier with the specified parameters."""
    logger.info("Starting yoga pose classifier training")
    
    # Create results directory
    run_dir, plots_dir = create_results_directory(args)
    
    # Initialize classifier
    classifier = YogaPoseClassifier()
    
    # Prepare dataset
    dataset_path = args.dataset
    logger.info(f"Using dataset from {dataset_path}")
    
    # Check for train/val/test splits
    if args.use_splits:
        train_path = os.path.join(dataset_path, "train")
        if os.path.exists(train_path):
            logger.info("Using existing train/val/test splits")
            dataset_path = train_path
        else:
            logger.warning("No train split found, using full dataset")
    
    try:
        # Extract features from dataset
        logger.info("Extracting features from dataset images")
        X, y = classifier.prepare_dataset_from_images(dataset_path)
        logger.info(f"Dataset prepared: {len(X)} samples, {len(np.unique(y))} classes")
        
        if len(X) == 0:
            logger.error("No valid samples extracted from dataset. Check images and directory structure.")
            return
        
        # Train model with appropriate algorithms
        logger.info("Training model")
        results = classifier.train(X, y)
        
        # Save trained model
        model_path = run_dir / f"{args.model_name}.pkl"
        classifier.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Generate visualizations if not disabled
        if not args.no_plots:
            visualize_training_results(results, classifier, plots_dir)
        
        # Save training summary
        save_training_summary(results, classifier, run_dir)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Test Accuracy: {results['test_accuracy']:.3f}")
        
        # Copy the model to a fixed path for the application to use
        production_model_path = Path("yoga_ai_trainer/backend/models/production_model.pkl")
        if results['test_accuracy'] >= 0.85:
            import shutil
            shutil.copy(model_path, production_model_path)
            logger.info(f"Model accuracy >= 85%, copied to production path: {production_model_path}")
        
        return classifier, results
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return None, None

def main():
    """Main function."""
    args = parse_arguments()
    classifier, results = train_model(args)
    
    if classifier and results:
        print("\n🎉 Training completed successfully!")
        print(f"🎯 Test Accuracy: {results['test_accuracy']:.3f}")
        if results['test_accuracy'] >= 0.85:
            print("✅ Model meets production accuracy threshold (>=85%)")
        else:
            print(f"⚠️ Model accuracy below production threshold. Target: 85%, Achieved: {results['test_accuracy']:.1f}%")
        print(f"📂 Results saved to {args.output}")
    else:
        print("\n❌ Training failed. Check the logs for details.")

if __name__ == "__main__":
    main()
