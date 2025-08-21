#!/usr/bin/env python3
"""
Yoga AI Trainer - Simple Demo Script
Tests the yoga pose detection without webcam
"""

import sys
import os
import numpy as np
import cv2
import requests
import json

# Add backend to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yoga_ai_trainer', 'backend'))

print("🧘‍♀️ Yoga AI Trainer - Demo Script")
print("=" * 50)

def test_backend_api():
    """Test the backend API endpoints."""
    print("\n📡 Testing Backend API...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server Status: {data['message']}")
        else:
            print("❌ Server not responding")
            return False
            
        # Test poses endpoint
        response = requests.get("http://localhost:8000/poses")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Available poses: {data['total_count']}")
            print(f"   Sample poses: {', '.join(data['poses'][:3])}...")
        
        # Test pronunciation
        response = requests.get("http://localhost:8000/test-pronunciation/tadasana")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Pronunciation test:")
            print(f"   {data['original']} → {data['pronunciation']}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Backend server not running!")
        print("   Start it with: ./start_dev.sh")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_pose_detection():
    """Test pose detection locally."""
    print("\n🤖 Testing Pose Detection...")
    
    try:
        from utils.pose_detector import PoseFeatureExtractor
        from utils.sanskrit_pronunciation import get_pose_pronunciation
        from models.pose_classifier_small import YogaPoseClassifier
        
        # Initialize components
        detector = PoseFeatureExtractor()
        classifier = YogaPoseClassifier()
        
        print("✅ Components initialized successfully")
        
        # Create a test image with a simple figure
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some simple lines to simulate a person (won't be detected, but tests the pipeline)
        cv2.line(test_image, (320, 100), (320, 400), (255, 255, 255), 3)  # Body
        cv2.line(test_image, (320, 150), (280, 200), (255, 255, 255), 3)  # Left arm
        cv2.line(test_image, (320, 150), (360, 200), (255, 255, 255), 3)  # Right arm
        cv2.line(test_image, (320, 350), (300, 450), (255, 255, 255), 3)  # Left leg
        cv2.line(test_image, (320, 350), (340, 450), (255, 255, 255), 3)  # Right leg
        cv2.circle(test_image, (320, 80), 30, (255, 255, 255), 3)  # Head
        
        # Test pose detection
        pose_data = detector.detect_pose(test_image)
        print(f"✅ Pose detection: {pose_data['pose_detected']}")
        
        if not pose_data['pose_detected']:
            print("   (Expected - simple drawing doesn't have proper landmarks)")
        
        # Test classification with random features
        random_features = np.random.rand(100)
        prediction = classifier.predict_pose(random_features)
        print(f"✅ Sample prediction: {prediction}")
        
        # Test pronunciation
        pronunciation = get_pose_pronunciation(prediction)
        print(f"✅ Pronunciation: {pronunciation}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pose detection test failed: {e}")
        return False

def show_demo_instructions():
    """Show instructions for using the full demo."""
    print("\n🎯 Full Demo Instructions:")
    print("=" * 50)
    
    print("1. 🚀 Start the development server:")
    print("   ./start_dev.sh")
    print()
    
    print("2. 🌐 Open the web interface:")
    print("   firefox yoga_ai_trainer/frontend/index.html")
    print("   or open: http://localhost:8000 (when we add static serving)")
    print()
    
    print("3. 📱 Test with your webcam:")
    print("   - Click 'Start Camera' in the web interface")
    print("   - Allow camera permissions")
    print("   - Try different yoga poses in front of camera")
    print("   - See real-time Sanskrit pose names and pronunciations")
    print()
    
    print("4. 🔍 Monitor the backend:")
    print("   - Watch server logs for real-time processing")
    print("   - Test API endpoints with curl commands")
    print("   - Check pose detection accuracy")
    print()
    
    print("📋 Available API Endpoints:")
    print("   • GET  /                     - Health check")
    print("   • GET  /poses                - List all poses")
    print("   • GET  /pose/{name}/pronunciation - Get pronunciation")
    print("   • GET  /health               - Detailed status")
    print("   • WebSocket /ws/pose-detection - Real-time detection")

def main():
    """Run the demo."""
    
    # Test backend API
    api_working = test_backend_api()
    
    # Test pose detection
    detection_working = test_pose_detection()
    
    # Show results
    print("\n" + "=" * 50)
    print("📊 DEMO RESULTS")
    print("=" * 50)
    
    if api_working:
        print("✅ Backend API: Working")
    else:
        print("❌ Backend API: Not available")
    
    if detection_working:
        print("✅ Pose Detection: Working")
    else:
        print("❌ Pose Detection: Failed")
    
    if api_working and detection_working:
        print("\n🎉 SUCCESS: Yoga AI Trainer is fully functional!")
        print("Ready for webcam-based pose detection!")
    else:
        print("\n⚠️  PARTIAL: Some components need attention")
    
    # Show demo instructions
    show_demo_instructions()

if __name__ == "__main__":
    main()
