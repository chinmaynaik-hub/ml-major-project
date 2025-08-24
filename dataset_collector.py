#!/usr/bin/env python3
"""
Yoga AI Trainer - Dataset Collection System
Automated collection and management of yoga pose images for training
"""

import os
import requests
from pathlib import Path
import json
import time
from typing import List, Dict
from PIL import Image
import hashlib
import logging
from urllib.parse import urljoin, urlparse
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YogaDatasetCollector:
    """
    Collect and manage yoga pose datasets for training
    """
    
    def __init__(self, dataset_root: str = "yoga_ai_trainer/backend/data/raw/yoga_poses/dataset"):
        self.dataset_root = Path(dataset_root)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        
        # Priority poses for initial training (based on common beginner poses)
        self.priority_poses = [
            "tadasana",           # Mountain Pose
            "vriksasana",         # Tree Pose
            "uttanasana",         # Forward Fold
            "adho_mukha_svanasana", # Downward Dog
            "bhujangasana",       # Cobra Pose
            "balasana",           # Child's Pose
            "trikonasana",        # Triangle Pose
            "parsvakonasana",     # Extended Side Angle
            "utthita_hasta_padangusthasana", # Extended Hand-to-Big-Toe
            "shavasana"           # Corpse Pose
        ]
        
        # Target images per pose for good training data
        self.target_images_per_pose = 50
        
        logger.info(f"Dataset collector initialized. Root: {self.dataset_root}")
    
    def create_dataset_structure(self):
        """Create the proper directory structure for the dataset"""
        logger.info("Creating dataset structure...")
        
        for pose in self.priority_poses:
            pose_dir = self.dataset_root / pose
            pose_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for organization
            (pose_dir / "raw").mkdir(exist_ok=True)
            (pose_dir / "processed").mkdir(exist_ok=True)
            (pose_dir / "validated").mkdir(exist_ok=True)
            
            logger.info(f"Created directories for {pose}")
        
        # Create metadata file
        self._create_dataset_metadata()
        
        logger.info("Dataset structure created successfully!")
    
    def _create_dataset_metadata(self):
        """Create metadata file with pose information"""
        metadata = {
            "dataset_info": {
                "name": "Yoga AI Trainer Dataset",
                "version": "1.0",
                "description": "Curated yoga pose images for pose classification training",
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "target_poses": len(self.priority_poses),
                "target_images_per_pose": self.target_images_per_pose
            },
            "poses": {}
        }
        
        # Add detailed pose information
        pose_info = {
            "tadasana": {
                "sanskrit": "ताड़ासन",
                "english": "Mountain Pose",
                "difficulty": 1,
                "category": "standing",
                "description": "Basic standing pose, foundation for all poses"
            },
            "vriksasana": {
                "sanskrit": "वृक्षासन",
                "english": "Tree Pose",
                "difficulty": 2,
                "category": "standing_balance",
                "description": "Single-leg balance pose"
            },
            "uttanasana": {
                "sanskrit": "उत्तानासन",
                "english": "Forward Fold",
                "difficulty": 2,
                "category": "forward_fold",
                "description": "Standing forward bend"
            },
            "adho_mukha_svanasana": {
                "sanskrit": "अधो मुख श्वानासन",
                "english": "Downward Facing Dog",
                "difficulty": 2,
                "category": "inversion",
                "description": "Inverted V-shape pose"
            },
            "bhujangasana": {
                "sanskrit": "भुजंगासन",
                "english": "Cobra Pose",
                "difficulty": 3,
                "category": "backbend",
                "description": "Gentle backbend, chest opener"
            },
            "balasana": {
                "sanskrit": "बालासन",
                "english": "Child's Pose",
                "difficulty": 1,
                "category": "restorative",
                "description": "Resting pose, forward fold on knees"
            },
            "trikonasana": {
                "sanskrit": "त्रिकोणासन",
                "english": "Triangle Pose",
                "difficulty": 2,
                "category": "standing",
                "description": "Side stretch with straight legs"
            },
            "parsvakonasana": {
                "sanskrit": "पार्श्वकोणासन",
                "english": "Extended Side Angle",
                "difficulty": 3,
                "category": "standing",
                "description": "Deep side stretch with arm extension"
            },
            "utthita_hasta_padangusthasana": {
                "sanskrit": "उत्थित हस्त पादांगुष्ठासन",
                "english": "Extended Hand-to-Big-Toe",
                "difficulty": 4,
                "category": "standing_balance",
                "description": "Standing balance with leg extension"
            },
            "shavasana": {
                "sanskrit": "शवासन",
                "english": "Corpse Pose",
                "difficulty": 1,
                "category": "restorative",
                "description": "Final relaxation pose"
            }
        }
        
        for pose in self.priority_poses:
            if pose in pose_info:
                metadata["poses"][pose] = pose_info[pose]
                metadata["poses"][pose]["collected_images"] = 0
                metadata["poses"][pose]["validated_images"] = 0
        
        # Save metadata
        metadata_file = self.dataset_root / "dataset_metadata.json"
        with metadata_file.open("w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info("Dataset metadata created")
    
    def validate_images(self, pose_name: str = None):
        """Validate collected images for quality and pose accuracy"""
        logger.info(f"Validating images for {pose_name or 'all poses'}...")
        
        poses_to_validate = [pose_name] if pose_name else self.priority_poses
        validated_count = 0
        
        for pose in poses_to_validate:
            pose_dir = self.dataset_root / pose
            raw_dir = pose_dir / "raw"
            validated_dir = pose_dir / "validated"
            
            if not raw_dir.exists():
                continue
            
            for img_file in raw_dir.glob("*.*"):
                if self._validate_single_image(img_file):
                    # Copy to validated directory
                    target = validated_dir / img_file.name
                    target.write_bytes(img_file.read_bytes())
                    validated_count += 1
                    logger.info(f"Validated: {pose}/{img_file.name}")
                else:
                    logger.warning(f"Failed validation: {pose}/{img_file.name}")
        
        logger.info(f"Validated {validated_count} images")
        return validated_count
    
    def _validate_single_image(self, img_path: Path) -> bool:
        """Validate a single image for basic quality requirements"""
        try:
            with Image.open(img_path) as img:
                # Check minimum dimensions
                if img.width < 224 or img.height < 224:
                    return False
                
                # Check aspect ratio (should be reasonable)
                aspect_ratio = img.width / img.height
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    return False
                
                # Check file size (not too small or too large)
                file_size = img_path.stat().st_size
                if file_size < 5000 or file_size > 5000000:  # 5KB - 5MB
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error validating {img_path}: {e}")
            return False
    
    def get_dataset_status(self) -> Dict:
        """Get current status of the dataset"""
        status = {
            "poses": {},
            "total_raw_images": 0,
            "total_validated_images": 0,
            "completion_percentage": 0
        }
        
        for pose in self.priority_poses:
            pose_dir = self.dataset_root / pose
            
            raw_count = 0
            validated_count = 0
            
            if (pose_dir / "raw").exists():
                raw_count = len(list((pose_dir / "raw").glob("*.*")))
            
            if (pose_dir / "validated").exists():
                validated_count = len(list((pose_dir / "validated").glob("*.*")))
            
            status["poses"][pose] = {
                "raw_images": raw_count,
                "validated_images": validated_count,
                "target": self.target_images_per_pose,
                "completion": min(validated_count / self.target_images_per_pose * 100, 100)
            }
            
            status["total_raw_images"] += raw_count
            status["total_validated_images"] += validated_count
        
        # Overall completion
        total_target = len(self.priority_poses) * self.target_images_per_pose
        status["completion_percentage"] = status["total_validated_images"] / total_target * 100
        
        return status
    
    def create_training_splits(self, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05):
        """Create training, validation, and test splits"""
        logger.info("Creating training splits...")
        
        # Create split directories
        for split in ["train", "val", "test"]:
            split_dir = self.dataset_root / split
            split_dir.mkdir(exist_ok=True)
            for pose in self.priority_poses:
                (split_dir / pose).mkdir(exist_ok=True)
        
        for pose in self.priority_poses:
            validated_dir = self.dataset_root / pose / "validated"
            
            if not validated_dir.exists():
                continue
            
            images = list(validated_dir.glob("*.*"))
            if not images:
                continue
            
            # Shuffle and split
            import random
            random.shuffle(images)
            
            n_images = len(images)
            n_train = int(n_images * train_ratio)
            n_val = int(n_images * val_ratio)
            
            # Copy images to appropriate splits
            for i, img_path in enumerate(images):
                if i < n_train:
                    split = "train"
                elif i < n_train + n_val:
                    split = "val"
                else:
                    split = "test"
                
                target = self.dataset_root / split / pose / img_path.name
                target.write_bytes(img_path.read_bytes())
            
            logger.info(f"{pose}: {n_train} train, {n_val} val, {n_images - n_train - n_val} test")
        
        logger.info("Training splits created successfully!")
    
    def generate_collection_guide(self):
        """Generate a guide for manual image collection"""
        guide_content = """
# Yoga Pose Dataset Collection Guide

## Overview
This guide will help you collect high-quality yoga pose images for training the AI model.

## Collection Requirements

### Image Quality Standards:
- **Resolution**: Minimum 224x224 pixels (prefer 512x512 or higher)
- **Format**: JPG, JPEG, or PNG
- **Size**: 5KB - 5MB per image
- **Lighting**: Good, even lighting
- **Background**: Varied backgrounds (studio, outdoor, home)
- **Person visibility**: Full body should be visible and clear

### Pose Diversity:
For each pose, collect images with:
- Different people (age, gender, body type diversity)
- Different angles (front view, side view, 3/4 view)
- Different variations (beginner to advanced modifications)
- Different clothing and settings
- Both left and right sides (where applicable)

## Priority Poses (50 images each):

"""
        
        # Add pose details
        metadata_file = self.dataset_root / "dataset_metadata.json"
        if metadata_file.exists():
            with metadata_file.open() as f:
                metadata = json.load(f)
            
            for pose_name, pose_info in metadata["poses"].items():
                guide_content += f"""
### {pose_info['english']} ({pose_info['sanskrit']})
- **Folder**: `{pose_name}/raw/`
- **Difficulty**: {pose_info['difficulty']}/5
- **Category**: {pose_info['category']}
- **Description**: {pose_info['description']}
- **Target Images**: 50

"""
        
        guide_content += """
## Collection Process:

1. **Create folder structure** (if not already done):
   ```bash
   python dataset_collector.py --create-structure
   ```

2. **Collect images manually**:
   - Save images to `dataset/{pose_name}/raw/` folders
   - Use descriptive filenames (e.g., `tadasana_01_front_view.jpg`)

3. **Validate collected images**:
   ```bash
   python dataset_collector.py --validate
   ```

4. **Check collection status**:
   ```bash
   python dataset_collector.py --status
   ```

5. **Create training splits** (when collection is complete):
   ```bash
   python dataset_collector.py --create-splits
   ```

## Image Sources:
- Personal photography (with permission)
- Yoga instruction websites (check usage rights)
- Stock photo sites (with proper licenses)
- Yoga community contributions
- Video frame extraction (from yoga videos)

## Quality Checklist:
- [ ] Person fully visible
- [ ] Pose clearly defined
- [ ] Good lighting
- [ ] Minimal blur/distortion
- [ ] Appropriate resolution
- [ ] Correct pose classification

## Next Steps:
Once you have 30+ validated images per pose, you can:
1. Start training the model with collected data
2. Iterate and improve based on model performance
3. Collect additional poses for expanded functionality
"""
        
        guide_file = self.dataset_root / "COLLECTION_GUIDE.md"
        guide_file.write_text(guide_content)
        
        logger.info(f"Collection guide created: {guide_file}")
        print(f"\n📖 Collection guide available at: {guide_file}")
        print("📁 Dataset structure ready for image collection!")

def main():
    parser = argparse.ArgumentParser(description="Yoga Dataset Collection System")
    parser.add_argument("--create-structure", action="store_true", 
                       help="Create the dataset directory structure")
    parser.add_argument("--validate", metavar="POSE", nargs="?", const="all",
                       help="Validate images for specified pose (or all poses)")
    parser.add_argument("--status", action="store_true",
                       help="Show dataset collection status")
    parser.add_argument("--create-splits", action="store_true",
                       help="Create training/validation/test splits")
    parser.add_argument("--guide", action="store_true",
                       help="Generate collection guide")
    
    args = parser.parse_args()
    
    collector = YogaDatasetCollector()
    
    if args.create_structure:
        collector.create_dataset_structure()
        collector.generate_collection_guide()
    
    elif args.validate:
        pose_name = None if args.validate == "all" else args.validate
        collector.validate_images(pose_name)
    
    elif args.status:
        status = collector.get_dataset_status()
        print("\n🧘‍♀️ Yoga Dataset Collection Status")
        print("=" * 50)
        print(f"📊 Overall Progress: {status['completion_percentage']:.1f}%")
        print(f"📁 Total Raw Images: {status['total_raw_images']}")
        print(f"✅ Total Validated Images: {status['total_validated_images']}")
        print()
        
        for pose, info in status["poses"].items():
            print(f"📝 {pose}:")
            print(f"   Raw: {info['raw_images']}, Validated: {info['validated_images']}/{info['target']} ({info['completion']:.1f}%)")
    
    elif args.create_splits:
        collector.create_training_splits()
    
    elif args.guide:
        collector.generate_collection_guide()
    
    else:
        print("🧘‍♀️ Yoga Dataset Collector")
        print("Usage examples:")
        print("  python dataset_collector.py --create-structure  # Set up directories")
        print("  python dataset_collector.py --status           # Check progress")
        print("  python dataset_collector.py --validate         # Validate all images")
        print("  python dataset_collector.py --create-splits    # Create train/val/test splits")
        print("  python dataset_collector.py --guide           # Generate collection guide")

if __name__ == "__main__":
    main()
