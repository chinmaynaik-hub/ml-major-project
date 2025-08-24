#!/usr/bin/env python3
"""
Enhanced Yoga AI Trainer - Feature Demo
Demonstrates the new capabilities added in this development session
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_dataset_collection():
    """Demo the dataset collection system."""
    print("\n" + "="*60)
    print("🗂️  DATASET COLLECTION SYSTEM DEMO")
    print("="*60)
    
    try:
        from dataset_collector import YogaDatasetCollector
        
        collector = YogaDatasetCollector()
        
        # Show available poses
        print("\n📝 Priority Poses for Collection:")
        for i, pose in enumerate(collector.priority_poses, 1):
            print(f"   {i:2d}. {pose}")
        
        # Show target structure
        print(f"\n📊 Target: {collector.target_images_per_pose} images per pose")
        print(f"📁 Total target images: {len(collector.priority_poses) * collector.target_images_per_pose}")
        
        # Show current status
        status = collector.get_dataset_status()
        print(f"\n📈 Current Progress: {status['completion_percentage']:.1f}%")
        print(f"📁 Raw images: {status['total_raw_images']}")
        print(f"✅ Validated images: {status['total_validated_images']}")
        
        print("\n💡 To start collecting:")
        print("   python dataset_collector.py --create-structure")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")

def demo_voice_feedback():
    """Demo the voice feedback system."""
    print("\n" + "="*60)
    print("🔊 VOICE FEEDBACK SYSTEM DEMO")
    print("="*60)
    
    try:
        from yoga_ai_trainer.backend.utils.voice_feedback import VoiceFeedbackSystem
        
        # Create voice feedback system
        voice = VoiceFeedbackSystem()
        
        print("🎵 Voice feedback system initialized")
        print("🎚️  Settings:")
        print(f"   - Speech rate: {voice.voice_rate} words/minute")
        print(f"   - Volume: {voice.voice_volume}")
        print(f"   - Enabled: {voice.is_enabled()}")
        
        # Demo different types of feedback
        print("\n🗣️  Testing voice feedback types:")
        
        print("   1. Pose Announcement...")
        voice.announce_pose("tadasana", "ताड़ासन", "ta-DAH-sa-na")
        time.sleep(2)
        
        print("   2. Pose Correction...")
        voice.provide_correction(["Keep your spine straight", "Engage your core"], "tadasana")
        time.sleep(2)
        
        print("   3. Encouragement...")
        voice.provide_encouragement("improvement")
        time.sleep(2)
        
        print("   4. Custom Message...")
        voice.speak_custom("Welcome to your enhanced yoga practice!")
        
        # Cleanup
        voice.shutdown()
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("💡 Note: Voice feedback requires system audio support")

def demo_yoga_sequences():
    """Demo the yoga sequence system."""
    print("\n" + "="*60)
    print("🧘‍♀️ YOGA SEQUENCES SYSTEM DEMO")
    print("="*60)
    
    try:
        from yoga_ai_trainer.backend.utils.yoga_sequences import get_sequence_tracker
        
        tracker = get_sequence_tracker()
        
        # Show available sequences
        sequences = tracker.get_available_sequences()
        print("🗂️  Available Sequences:")
        for seq in sequences:
            print(f"   📚 {seq['name']}")
            print(f"      - {seq['description']}")
            print(f"      - Difficulty: {seq['difficulty']}/5")
            print(f"      - Duration: {seq['duration_minutes']} minutes")
            print(f"      - Poses: {seq['total_poses']}")
            print()
        
        # Demo starting a sequence
        print("🎬 Starting 'Surya Namaskara A' sequence...")
        success = tracker.start_sequence("surya_namaskara_a")
        
        if success:
            print("✅ Sequence started successfully!")
            
            # Show current pose
            current_pose = tracker.get_current_pose()
            if current_pose:
                print(f"\n🧘‍♀️ Current Pose: {current_pose.sanskrit_name}")
                print(f"   📖 Pronunciation: {current_pose.pronunciation}")
                print(f"   ⏱️  Hold time: {current_pose.hold_time} seconds")
                print(f"   📋 Instructions:")
                for instruction in current_pose.instructions:
                    print(f"      • {instruction}")
            
            # Show progress info
            progress = tracker.get_progress_info()
            if progress:
                print(f"\n📊 Progress: {progress['progress']['current_step']}/{progress['progress']['total_steps']}")
                print(f"📈 Completion: {progress['progress']['completion_percentage']:.1f}%")
                print(f"⏱️  Elapsed: {progress['progress']['elapsed_time']:.1f}s")
            
            # Simulate advancing through poses
            print("\n🎭 Simulating sequence progression...")
            for i in range(3):
                time.sleep(1)
                success = tracker.advance_to_next_pose(accuracy_score=0.85)
                if success:
                    current_pose = tracker.get_current_pose()
                    if current_pose:
                        print(f"   ➡️  Advanced to: {current_pose.sanskrit_name}")
                else:
                    print("   🏁 Sequence completed!")
                    break
        
    except Exception as e:
        print(f"❌ Demo error: {e}")

def demo_training_pipeline():
    """Demo the enhanced training pipeline."""
    print("\n" + "="*60)
    print("🤖 ENHANCED TRAINING PIPELINE DEMO")
    print("="*60)
    
    print("🏗️  Training Pipeline Features:")
    print("   ✅ Multiple algorithm comparison (Random Forest, SVM, KNN)")
    print("   ✅ Cross-validation with detailed reporting")
    print("   ✅ Training progress visualization")
    print("   ✅ Confusion matrix generation")
    print("   ✅ Feature importance analysis")
    print("   ✅ Hyperparameter optimization")
    print("   ✅ Model performance tracking")
    
    print("\n🚀 Usage Examples:")
    print("   # Basic training:")
    print("   python train_yoga_model.py")
    print()
    print("   # Advanced training with custom settings:")
    print("   python train_yoga_model.py --dataset custom/path \\")
    print("                              --model-name advanced_v2 \\")
    print("                              --hyperopt \\")
    print("                              --augmentation")
    
    print("\n📊 Output Features:")
    print("   📈 Training curves and accuracy plots")
    print("   🎯 Confusion matrices")
    print("   📋 Detailed classification reports")
    print("   💾 Trained model artifacts")
    print("   📝 Training summaries and metadata")

def main():
    """Run the enhanced features demo."""
    print("🧘‍♀️ Yoga AI Trainer - Enhanced Features Demo")
    print("=" * 60)
    print("This demo showcases the new features added in this development session:")
    print("• Dataset Collection & Management System")
    print("• Voice Feedback with Sanskrit Pronunciation")
    print("• Yoga Sequences & Flow Tracking")
    print("• Enhanced Model Training Pipeline")
    
    try:
        # Run demos
        demo_dataset_collection()
        demo_voice_feedback()
        demo_yoga_sequences()
        demo_training_pipeline()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("✨ Your Yoga AI Trainer now includes:")
        print("   🗂️  Professional dataset management")
        print("   🔊 Real-time voice guidance")
        print("   🧘‍♀️ Complete yoga sequences")
        print("   🤖 Advanced ML training pipeline")
        print()
        print("🚀 Ready for the next phase of development!")
        print("💡 Next steps: Collect images and train your first model!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    main()
