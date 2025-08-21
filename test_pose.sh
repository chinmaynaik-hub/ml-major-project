#!/bin/bash

# Yoga Pose Detection Test Script
# Usage: ./test_pose.sh <image_path>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_path>"
    echo ""
    echo "Examples:"
    echo "  $0 /home/chinmay/Downloads/yoga_test_image.jpg"
    echo "  $0 ./my_yoga_pose.png"
    echo "  $0 /path/to/any/image.jpeg"
    exit 1
fi

IMAGE_PATH="$1"

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file '$IMAGE_PATH' not found!"
    exit 1
fi

echo "🧘 Testing pose detection with: $IMAGE_PATH"
echo "📤 Sending image to backend..."
echo "----------------------------------------"

# Send image to pose detection API
response=$(curl -s -X POST \
     -F "file=@$IMAGE_PATH" \
     http://localhost:8000/detect-pose)

# Check if request was successful
if [ $? -eq 0 ]; then
    echo "✅ Response received!"
    echo "$response" | python3 -m json.tool
else
    echo "❌ Error: Could not connect to backend. Make sure it's running on localhost:8000"
fi

echo ""
echo "----------------------------------------"
echo "🏁 Test completed!"
