#!/usr/bin/env python3
"""
Test script for Yoga AI Trainer pose detection
Verifies MediaPipe integration and basic functionality
"""

import sys
import os
import numpy as np
import cv2

# Add backend to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yoga_ai_trainer', 'backend'))

try:
    from utils.pose_detector import PoseFeatureExtractor
    from utils.sanskrit_pronunciation import get_pose_pronunciation, generate_voice_text
    from models.pose_classifier_small import YogaPoseClassifier
    print("✓ All modules imported successfully!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def test_pose_detector():
    """Test the pose feature extractor."""
    print("\n🧘‍♀️ Testing Pose Feature Extractor...")
    
    try:
        detector = PoseFeatureExtractor()
        print("✓ PoseFeatureExtractor initialized")
        
        # Create a dummy image for testing (black image)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test pose detection
        pose_data = detector.detect_pose(dummy_image)
        print(f"✓ Pose detection completed. Pose detected: {pose_data['pose_detected']}")
        
        if pose_data['pose_detected']:
            print(f"  - Confidence: {pose_data['confidence']:.3f}")
            print(f"  - Number of keypoints: {len(pose_data['keypoints'])}")
            print(f"  - Number of angles: {len(pose_data['angles'])}")
            
            # Test feature extraction
            features = detector.get_pose_features(pose_data['keypoints'], pose_data['angles'])
            print(f"  - Feature vector shape: {features.shape}")
        else:
            print("  - No pose detected in dummy image (expected)")
            
        return True
        
    except Exception as e:
        print(f"✗ PoseFeatureExtractor test failed: {e}")
        return False

def test_sanskrit_pronunciation():
    """Test Sanskrit pronunciation utilities."""
    print("\n📢 Testing Sanskrit Pronunciation...")
    
    try:
        # Test pose names
        test_poses = ["tadasana", "vriksasana", "bhujangasana", "unknown_pose"]
        
        for pose in test_poses:
            pronunciation = get_pose_pronunciation(pose)
            voice_text = generate_voice_text(pose, "Great pose!")
            
            print(f"  - {pose}: {pronunciation}")
            print(f"    Voice: {voice_text}")
        
        print("✓ Sanskrit pronunciation test completed")
        return True
        
    except Exception as e:
        print(f"✗ Sanskrit pronunciation test failed: {e}")
        return False

def test_pose_classifier():
    """Test the yoga pose classifier."""
    print("\n🤖 Testing Yoga Pose Classifier...")
    
    try:
        classifier = YogaPoseClassifier()
        print("✓ YogaPoseClassifier initialized")
        
        # Test getting pose names
        pose_names = classifier.get_pose_names()
        print(f"  - Available poses: {pose_names}")
        
        # Test simple prediction (without trained model)
        dummy_landmarks = np.random.rand(100)  # Random feature vector
        prediction = classifier.predict_pose(dummy_landmarks)
        print(f"  - Test prediction: {prediction}")
        
        print("✓ YogaPoseClassifier test completed")
        return True
        
    except Exception as e:
        print(f"✗ YogaPoseClassifier test failed: {e}")
        return False

def test_mediapipe_basics():
    """Test basic MediaPipe functionality."""
    print("\n🎥 Testing MediaPipe Basics...")
    
    try:
        import mediapipe as mp
        
        # Initialize MediaPipe pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✓ MediaPipe pose initialized")
        
        # Test with a simple image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
        results = pose.process(rgb_image)
        print(f"✓ MediaPipe processing completed. Landmarks detected: {results.pose_landmarks is not None}")
        
        return True
        
    except Exception as e:
        print(f"✗ MediaPipe test failed: {e}")
        return False

def test_dependencies():
    """Test all required dependencies."""
    print("\n📦 Testing Dependencies...")
    
    required_modules = [
        'numpy', 'opencv-python', 'mediapipe', 'scikit-learn', 
        'fastapi', 'uvicorn', 'pillow'
    ]
    
    missing_modules = []
    
    for module_name in required_modules:
        try:
            # Handle special cases
            if module_name == 'opencv-python':
                import cv2
            elif module_name == 'pillow':
                import PIL
            elif module_name == 'scikit-learn':
                import sklearn
            else:
                __import__(module_name.replace('-', '_'))
            print(f"✓ {module_name}")
        except ImportError:
            print(f"✗ {module_name} - MISSING")
            missing_modules.append(module_name)
    
    if missing_modules:
        print(f"\n❌ Missing dependencies: {', '.join(missing_modules)}")
        print("Install with: pip install " + " ".join(missing_modules))
        return False
    else:
        print("✓ All dependencies available")
        return True

def main():
    """Run all tests."""
    print("🧘‍♀️ Yoga AI Trainer - System Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("MediaPipe Basics", test_mediapipe_basics),
        ("Sanskrit Pronunciation", test_sanskrit_pronunciation),
        ("Pose Detector", test_pose_detector),
        ("Pose Classifier", test_pose_classifier)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print("\n\n⚠️ Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status:<8} {test_name}")
        if passed_test:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Yoga AI Trainer is ready for development.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
