#!/usr/bin/env python3

import requests
import json
import sys
import os

def test_pose_detection(image_path):
    """Test the pose detection API with a given image"""
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' does not exist")
        return
    
    # API endpoint
    url = "http://localhost:8000/detect-pose"
    
    try:
        # Prepare the file for upload
        with open(image_path, 'rb') as f:
            files = {'file': f}
            
            print(f"🔄 Testing pose detection with: {image_path}")
            print("📡 Sending request to API...")
            
            # Send POST request
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                # Parse JSON response
                result = response.json()
                
                print("✅ Pose Detection Results:")
                print("=" * 50)
                print(f"📝 Pose Detected: {result.get('pose', 'Unknown')}")
                print(f"🗣️  Pronunciation: {result.get('pronunciation', 'N/A')}")
                print(f"🔊 Voice Feedback: {result.get('voice_feedback', 'N/A')}")
                
                # Check if annotated image is included
                if 'annotated_image' in result:
                    print("🖼️  Annotated image: Included in response (base64 encoded)")
                else:
                    print("🖼️  Annotated image: Not included")
                    
                print("=" * 50)
                print("✅ Test completed successfully!")
                
            else:
                print(f"❌ API Error: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server at localhost:8000")
        print("Make sure the backend server is running!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 test_pose.py <image_path>")
        print("Example: python3 test_pose.py /home/chinmay/Downloads/yoga_test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_pose_detection(image_path)

if __name__ == "__main__":
    main()
