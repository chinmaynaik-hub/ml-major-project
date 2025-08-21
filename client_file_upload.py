#!/usr/bin/env python3
"""
Yoga AI Trainer - File Upload Client
Send images to backend via file upload
"""

import requests
import json
import os
from pathlib import Path

def test_image_upload(image_path: str, server_url: str = "http://localhost:8000"):
    """
    Upload an image file to the backend for pose detection
    
    Args:
        image_path: Path to the image file
        server_url: Backend server URL
    """
    print(f"🖼️ Uploading image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    try:
        # Open the image file
        with open(image_path, 'rb') as image_file:
            files = {'file': (os.path.basename(image_path), image_file, 'image/jpeg')}
            
            # Send POST request
            response = requests.post(f"{server_url}/detect-pose", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Upload successful!")
                print(f"📁 Filename: {result.get('filename', 'unknown')}")
                print(f"🧘‍♀️ Pose detected: {result.get('pose_detected', False)}")
                
                if result.get('detected_pose'):
                    print(f"🕉️ Detected pose: {result['detected_pose']}")
                    print(f"🗣️ Pronunciation: {result.get('pronunciation', 'N/A')}")
                    print(f"💬 Voice text: {result.get('voice_text', 'N/A')}")
                else:
                    print("ℹ️ No pose detected in the image")
                
                return result
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"Error: {response.text}")
                return None
                
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend server!")
        print("Start the server with: ./start_dev.sh")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def create_test_image():
    """Create a simple test image if no real image is available"""
    import cv2
    import numpy as np
    
    # Create a simple test image with a stick figure
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple stick figure
    cv2.circle(img, (320, 100), 30, (255, 255, 255), 3)  # Head
    cv2.line(img, (320, 130), (320, 350), (255, 255, 255), 3)  # Body
    cv2.line(img, (320, 180), (250, 220), (255, 255, 255), 3)  # Left arm
    cv2.line(img, (320, 180), (390, 220), (255, 255, 255), 3)  # Right arm
    cv2.line(img, (320, 350), (280, 450), (255, 255, 255), 3)  # Left leg
    cv2.line(img, (320, 350), (360, 450), (255, 255, 255), 3)  # Right leg
    
    # Save test image
    test_image_path = "test_pose.jpg"
    cv2.imwrite(test_image_path, img)
    print(f"📝 Created test image: {test_image_path}")
    return test_image_path

def main():
    print("🧘‍♀️ Yoga AI Trainer - File Upload Client")
    print("=" * 50)
    
    # Check if any image files exist in current directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path('.').glob(f'*{ext}'))
        image_files.extend(Path('.').glob(f'*{ext.upper()}'))
    
    if image_files:
        print("📁 Found image files:")
        for i, img_path in enumerate(image_files[:5]):  # Show max 5 files
            print(f"   {i+1}. {img_path}")
        
        # Test with the first image found
        test_image_path = str(image_files[0])
        print(f"\n🧪 Testing with: {test_image_path}")
        result = test_image_upload(test_image_path)
        
    else:
        print("📁 No image files found in current directory")
        print("🎨 Creating a test image...")
        test_image_path = create_test_image()
        result = test_image_upload(test_image_path)
    
    print("\n" + "=" * 50)
    print("📋 Usage Examples:")
    print("=" * 50)
    print("# Upload any image file:")
    print("python client_file_upload.py")
    print("\n# Or use curl command:")
    print("curl -X POST -F 'file=@your_image.jpg' http://localhost:8000/detect-pose")
    print("\n# Or use requests in Python:")
    print("import requests")
    print("with open('image.jpg', 'rb') as f:")
    print("    files = {'file': ('image.jpg', f, 'image/jpeg')}")
    print("    response = requests.post('http://localhost:8000/detect-pose', files=files)")
    print("    print(response.json())")

if __name__ == "__main__":
    main()
