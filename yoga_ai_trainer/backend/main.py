#!/usr/bin/env python3
"""
Yoga AI Trainer - Main FastAPI Backend
Real-time yoga pose detection with Sanskrit voice feedback
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from typing import List, Dict, Any
import json
import base64
from io import BytesIO
from PIL import Image
import logging
from pathlib import Path

# Import our custom modules
try:
    # Try relative imports first (when running from backend directory)
    from utils.sanskrit_pronunciation import get_pose_pronunciation, generate_voice_text
    from models.pose_classifier_small import YogaPoseClassifier
except ImportError:
    # Use absolute imports (when running from project root)
    import sys
    import os
    backend_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, backend_path)
    from utils.sanskrit_pronunciation import get_pose_pronunciation, generate_voice_text
    from models.pose_classifier_small import YogaPoseClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Yoga AI Trainer",
    description="AI-powered yoga trainer with Sanskrit pose names and real-time feedback",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize pose classifier
pose_classifier = YogaPoseClassifier()

# Global connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.get("/")
async def read_root():
    """Basic health check endpoint"""
    return {"message": "🧘‍♀️ Yoga AI Trainer is running!", "status": "healthy"}

@app.get("/poses")
async def get_available_poses():
    """Get list of all available yoga poses"""
    try:
        poses = pose_classifier.get_pose_names()
        return {
            "poses": poses,
            "total_count": len(poses),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting poses: {e}")
        return {"error": str(e), "status": "error"}

@app.get("/pose/{pose_name}/pronunciation")
async def get_pose_pronunciation_endpoint(pose_name: str):
    """Get Sanskrit pronunciation for a specific pose"""
    try:
        pronunciation = get_pose_pronunciation(pose_name)
        voice_text = generate_voice_text(pose_name)
        
        return {
            "pose_name": pose_name,
            "pronunciation": pronunciation,
            "voice_text": voice_text,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting pronunciation: {e}")
        return {"error": str(e), "status": "error"}

def process_pose_landmarks(landmarks):
    """Process MediaPipe landmarks for pose classification"""
    if not landmarks:
        return None
    
    # Extract landmark coordinates
    landmark_list = []
    for landmark in landmarks.landmark:
        landmark_list.extend([landmark.x, landmark.y, landmark.z])
    
    return np.array(landmark_list)

def detect_pose_from_frame(frame):
    """Detect pose from camera frame using MediaPipe"""
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = pose.process(rgb_frame)
        
        # Draw landmarks on the frame
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )
            
            # Extract landmarks for classification
            landmarks_array = process_pose_landmarks(results.pose_landmarks)
            
            if landmarks_array is not None:
                # Classify the pose (placeholder - will implement actual classification)
                detected_pose = pose_classifier.predict_pose(landmarks_array)
                return annotated_frame, detected_pose, results.pose_landmarks
        
        return annotated_frame, None, None
        
    except Exception as e:
        logger.error(f"Error in pose detection: {e}")
        return frame, None, None

@app.websocket("/ws/pose-detection")
async def websocket_pose_detection(websocket: WebSocket):
    """WebSocket endpoint for real-time pose detection"""
    await manager.connect(websocket)
    logger.info("WebSocket connected for pose detection")
    
    try:
        while True:
            # Receive image data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "frame":
                # Decode base64 image
                image_data = base64.b64decode(message["data"].split(",")[1])
                image = Image.open(BytesIO(image_data))
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Process the frame
                annotated_frame, detected_pose, landmarks = detect_pose_from_frame(frame)
                
                # Prepare response
                response = {
                    "type": "pose_detection",
                    "timestamp": message.get("timestamp"),
                    "detected_pose": detected_pose,
                    "has_landmarks": landmarks is not None
                }
                
                if detected_pose:
                    pronunciation = get_pose_pronunciation(detected_pose)
                    response["pronunciation"] = pronunciation
                    response["voice_text"] = generate_voice_text(detected_pose)
                
                # Encode annotated frame back to base64
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                img_str = base64.b64encode(buffer).decode()
                response["annotated_frame"] = f"data:image/jpeg;base64,{img_str}"
                
                await manager.send_personal_message(json.dumps(response), websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected from pose detection")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.send_personal_message(
            json.dumps({"type": "error", "message": str(e)}), 
            websocket
        )

@app.get("/test-pronunciation/{pose_name}")
async def test_pronunciation(pose_name: str):
    """Test endpoint to check Sanskrit pronunciation"""
    try:
        pronunciation = get_pose_pronunciation(pose_name)
        voice_text = generate_voice_text(pose_name, "Align your spine and breathe deeply.")
        
        return {
            "original": pose_name,
            "pronunciation": pronunciation,
            "with_feedback": voice_text,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

@app.post("/detect-pose")
async def detect_pose_from_image(file: UploadFile = File(...)):
    """Detect pose from uploaded image file"""
    try:
        # Read the uploaded image
        image_data = await file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Invalid image format", "status": "error"}
        
        # Process the frame
        annotated_frame, detected_pose, landmarks = detect_pose_from_frame(frame)
        
        # Prepare response
        response = {
            "status": "success",
            "pose_detected": landmarks is not None,
            "detected_pose": detected_pose,
            "filename": file.filename
        }
        
        if detected_pose:
            pronunciation = get_pose_pronunciation(detected_pose)
            response["pronunciation"] = pronunciation
            response["voice_text"] = generate_voice_text(detected_pose)
        
        # Optionally encode annotated frame back to base64
        if landmarks is not None:
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            img_str = base64.b64encode(buffer).decode()
            response["annotated_image"] = f"data:image/jpeg;base64,{img_str}"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in pose detection: {e}")
        return {"error": str(e), "status": "error"}

@app.post("/detect-pose-base64")
async def detect_pose_from_base64(image_data: str = Form(...)):
    """Detect pose from base64 encoded image"""
    try:
        # Decode base64 image
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process the frame
        annotated_frame, detected_pose, landmarks = detect_pose_from_frame(frame)
        
        # Prepare response
        response = {
            "status": "success",
            "pose_detected": landmarks is not None,
            "detected_pose": detected_pose
        }
        
        if detected_pose:
            pronunciation = get_pose_pronunciation(detected_pose)
            response["pronunciation"] = pronunciation
            response["voice_text"] = generate_voice_text(detected_pose)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in pose detection: {e}")
        return {"error": str(e), "status": "error"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test MediaPipe
        mp_status = "MediaPipe initialized" if pose else "MediaPipe not initialized"
        
        # Test pose classifier
        classifier_status = "Pose classifier loaded" if pose_classifier else "Pose classifier not loaded"
        
        # Check dataset
        dataset_path = Path("./yoga_ai_trainer/backend/data/raw/yoga_poses/dataset")
        dataset_status = f"Dataset found ({len(list(dataset_path.iterdir()))} poses)" if dataset_path.exists() else "Dataset not found"
        
        return {
            "status": "healthy",
            "mediapipe": mp_status,
            "pose_classifier": classifier_status,
            "dataset": dataset_status,
            "pronunciation_mappings": len(get_pose_pronunciation.__code__.co_consts) if hasattr(get_pose_pronunciation, '__code__') else "Unknown"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    
    # Start the server
    logger.info("🧘‍♀️ Starting Yoga AI Trainer Backend...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
