#!/usr/bin/env python3
"""
Script to download and organize the yoga pose dataset from Kaggle
"""

import kagglehub
import os
import shutil
from pathlib import Path

def download_and_organize_dataset():
    """Download the yoga pose dataset and organize it in the project structure"""
    
    print("🧘‍♀️ Downloading Yoga Pose Classification Dataset...")
    
    # Download latest version
    try:
        path = kagglehub.dataset_download("shrutisaxena/yoga-pose-image-classification-dataset")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Create organized data structure in the project
        project_data_dir = Path("./yoga_ai_trainer/backend/data")
        project_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (project_data_dir / "raw").mkdir(exist_ok=True)
        (project_data_dir / "processed").mkdir(exist_ok=True)
        (project_data_dir / "models").mkdir(exist_ok=True)
        
        # Copy dataset to project structure
        raw_data_dir = project_data_dir / "raw" / "yoga_poses"
        
        if raw_data_dir.exists():
            shutil.rmtree(raw_data_dir)
            
        shutil.copytree(path, raw_data_dir)
        print(f"✅ Dataset organized in: {raw_data_dir}")
        
        # List the contents to understand structure
        print("\n📁 Dataset structure:")
        for root, dirs, files in os.walk(raw_data_dir):
            level = root.replace(str(raw_data_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files per directory
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
        
        # Create a metadata file
        metadata_file = project_data_dir / "dataset_info.txt"
        with open(metadata_file, 'w') as f:
            f.write("Yoga Pose Image Classification Dataset\n")
            f.write("="*50 + "\n")
            f.write(f"Source: shrutisaxena/yoga-pose-image-classification-dataset\n")
            f.write(f"Downloaded to: {path}\n")
            f.write(f"Organized in: {raw_data_dir}\n")
            f.write(f"Download date: {os.popen('date').read().strip()}\n")
        
        print(f"✅ Metadata saved to: {metadata_file}")
        
        return str(raw_data_dir)
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    dataset_path = download_and_organize_dataset()
    if dataset_path:
        print(f"\n🎉 Success! Dataset is ready at: {dataset_path}")
        print("\nNext steps:")
        print("1. Explore the dataset structure")
        print("2. Create data preprocessing scripts")
        print("3. Set up the pose classification model")
    else:
        print("\n❌ Failed to download dataset. Please check your internet connection.")
