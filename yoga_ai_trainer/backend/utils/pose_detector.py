"""
Pose Detection Utilities for Yoga AI Trainer
Extracts features from MediaPipe pose landmarks for classification.
"""

import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import math


class PoseFeatureExtractor:
    """
    Extracts pose features from MediaPipe landmarks for yoga pose classification.
    
    Features extracted:
    - Joint angles (14 key angles)
    - Normalized landmark coordinates (33 landmarks × 2 = 66 coordinates)
    - Pose ratios and proportions (additional geometric features)
    """
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key body connections for angle calculation
        self.angle_connections = [
            # Upper body angles
            ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),     # Left arm
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),   # Right arm
            ('LEFT_HIP', 'LEFT_SHOULDER', 'LEFT_ELBOW'),        # Left shoulder-body
            ('RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),     # Right shoulder-body
            
            # Core angles
            ('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),         # Left side bend
            ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'),      # Right side bend
            
            # Lower body angles
            ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),            # Left leg
            ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),         # Right leg
            ('LEFT_KNEE', 'LEFT_HIP', 'RIGHT_HIP'),             # Left hip
            ('RIGHT_KNEE', 'RIGHT_HIP', 'LEFT_HIP'),            # Right hip
            
            # Spine and torso
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP'),    # Upper torso twist
            ('LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE'),             # Lower torso
            
            # Balance and stability
            ('LEFT_ANKLE', 'LEFT_KNEE', 'LEFT_HIP'),            # Left leg alignment
            ('RIGHT_ANKLE', 'RIGHT_KNEE', 'RIGHT_HIP'),         # Right leg alignment
        ]
        
        # Key landmarks for coordinate features
        self.key_landmarks = [
            'NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EAR', 'RIGHT_EAR',
            'MOUTH_LEFT', 'MOUTH_RIGHT',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
            'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_PINKY', 'RIGHT_PINKY',
            'LEFT_INDEX', 'RIGHT_INDEX',
            'LEFT_THUMB', 'RIGHT_THUMB',
            'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE',
            'LEFT_HEEL', 'RIGHT_HEEL',
            'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
        ]
    
    def detect_pose(self, image: np.ndarray) -> Dict:
        """
        Detect pose from image using MediaPipe.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Dictionary with pose detection results
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(rgb_image)
        
        if results.pose_landmarks:
            # Extract landmark coordinates
            landmarks = self._extract_landmark_coordinates(results.pose_landmarks)
            
            # Calculate angles
            angles = self._calculate_pose_angles(landmarks)
            
            # Calculate confidence (based on landmark visibility)
            confidence = self._calculate_pose_confidence(results.pose_landmarks)
            
            # Create annotated image
            annotated_image = image.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            return {
                'pose_detected': True,
                'keypoints': landmarks,
                'angles': angles,
                'confidence': confidence,
                'processed_image': annotated_image,
                'raw_landmarks': results.pose_landmarks
            }
        else:
            return {
                'pose_detected': False,
                'keypoints': None,
                'angles': None,
                'confidence': 0.0,
                'processed_image': image,
                'raw_landmarks': None
            }
    
    def _extract_landmark_coordinates(self, pose_landmarks) -> Dict[str, Tuple[float, float]]:
        """Extract normalized landmark coordinates."""
        landmarks = {}
        
        for landmark_name in self.key_landmarks:
            landmark_id = getattr(self.mp_pose.PoseLandmark, landmark_name)
            landmark = pose_landmarks.landmark[landmark_id]
            
            # Store normalized coordinates (x, y)
            landmarks[landmark_name] = (landmark.x, landmark.y)
        
        return landmarks
    
    def _calculate_pose_angles(self, landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Calculate joint angles from landmark coordinates."""
        angles = {}
        
        for i, (point1, point2, point3) in enumerate(self.angle_connections):
            try:
                # Get coordinates
                p1 = landmarks[point1]
                p2 = landmarks[point2]  # vertex of the angle
                p3 = landmarks[point3]
                
                # Calculate angle
                angle = self._calculate_angle(p1, p2, p3)
                angles[f'angle_{i}_{point1}_{point2}_{point3}'] = angle
                
            except KeyError:
                # If landmark is missing, set angle to 0
                angles[f'angle_{i}_{point1}_{point2}_{point3}'] = 0.0
        
        return angles
    
    def _calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """
        Calculate angle between three points.
        
        Args:
            p1, p2, p3: Points where p2 is the vertex
            
        Returns:
            Angle in degrees (0-180)
        """
        # Calculate vectors
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Calculate dot product and magnitudes
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        # Avoid division by zero
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0.0
        
        # Calculate cosine of angle
        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
        
        # Clamp to valid range for arccos
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        # Convert to degrees
        angle = math.acos(cos_angle) * 180 / math.pi
        
        return angle
    
    def _calculate_pose_confidence(self, pose_landmarks) -> float:
        """Calculate pose detection confidence based on landmark visibility."""
        visible_landmarks = 0
        total_landmarks = len(pose_landmarks.landmark)
        
        for landmark in pose_landmarks.landmark:
            if landmark.visibility > 0.5:  # MediaPipe visibility threshold
                visible_landmarks += 1
        
        return visible_landmarks / total_landmarks if total_landmarks > 0 else 0.0
    
    def get_pose_features(self, landmarks: Dict[str, Tuple[float, float]], angles: Dict[str, float]) -> np.ndarray:
        """
        Combine all pose features into a single feature vector.
        
        Args:
            landmarks: Landmark coordinates
            angles: Joint angles
            
        Returns:
            Feature vector for machine learning model
        """
        features = []
        
        # Add landmark coordinates (flattened)
        for landmark_name in self.key_landmarks:
            if landmark_name in landmarks:
                x, y = landmarks[landmark_name]
                features.extend([x, y])
            else:
                features.extend([0.0, 0.0])  # Missing landmark
        
        # Add angles
        for connection in self.angle_connections:
            angle_key = f'angle_{self.angle_connections.index(connection)}_{connection[0]}_{connection[1]}_{connection[2]}'
            if angle_key in angles:
                features.append(angles[angle_key] / 180.0)  # Normalize to 0-1
            else:
                features.append(0.0)  # Missing angle
        
        # Add geometric features
        geometric_features = self._calculate_geometric_features(landmarks)
        features.extend(geometric_features)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_geometric_features(self, landmarks: Dict[str, Tuple[float, float]]) -> List[float]:
        """
        Calculate additional geometric features for pose characterization.
        
        Features:
        - Body ratios (shoulder width, hip width, torso length, etc.)
        - Symmetry measures
        - Center of mass approximation
        """
        features = []
        
        try:
            # Shoulder width
            shoulder_width = abs(landmarks['LEFT_SHOULDER'][0] - landmarks['RIGHT_SHOULDER'][0])
            features.append(shoulder_width)
            
            # Hip width
            hip_width = abs(landmarks['LEFT_HIP'][0] - landmarks['RIGHT_HIP'][0])
            features.append(hip_width)
            
            # Torso length (shoulder to hip center)
            shoulder_center_x = (landmarks['LEFT_SHOULDER'][0] + landmarks['RIGHT_SHOULDER'][0]) / 2
            shoulder_center_y = (landmarks['LEFT_SHOULDER'][1] + landmarks['RIGHT_SHOULDER'][1]) / 2
            hip_center_x = (landmarks['LEFT_HIP'][0] + landmarks['RIGHT_HIP'][0]) / 2
            hip_center_y = (landmarks['LEFT_HIP'][1] + landmarks['RIGHT_HIP'][1]) / 2
            
            torso_length = math.sqrt((shoulder_center_x - hip_center_x)**2 + (shoulder_center_y - hip_center_y)**2)
            features.append(torso_length)
            
            # Left leg length (approximate)
            left_leg_length = math.sqrt(
                (landmarks['LEFT_HIP'][0] - landmarks['LEFT_ANKLE'][0])**2 + 
                (landmarks['LEFT_HIP'][1] - landmarks['LEFT_ANKLE'][1])**2
            )
            features.append(left_leg_length)
            
            # Right leg length (approximate)
            right_leg_length = math.sqrt(
                (landmarks['RIGHT_HIP'][0] - landmarks['RIGHT_ANKLE'][0])**2 + 
                (landmarks['RIGHT_HIP'][1] - landmarks['RIGHT_ANKLE'][1])**2
            )
            features.append(right_leg_length)
            
            # Symmetry measure (difference in leg lengths)
            leg_symmetry = abs(left_leg_length - right_leg_length)
            features.append(leg_symmetry)
            
            # Body center x-coordinate
            body_center_x = (shoulder_center_x + hip_center_x) / 2
            features.append(body_center_x)
            
        except KeyError:
            # If landmarks are missing, fill with zeros
            features = [0.0] * 7
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features for interpretability."""
        feature_names = []
        
        # Landmark coordinate names
        for landmark_name in self.key_landmarks:
            feature_names.extend([f'{landmark_name}_x', f'{landmark_name}_y'])
        
        # Angle names
        for i, (point1, point2, point3) in enumerate(self.angle_connections):
            feature_names.append(f'angle_{i}_{point1}_{point2}_{point3}')
        
        # Geometric feature names
        geometric_names = [
            'shoulder_width', 'hip_width', 'torso_length', 
            'left_leg_length', 'right_leg_length', 'leg_symmetry', 'body_center_x'
        ]
        feature_names.extend(geometric_names)
        
        return feature_names


# Example usage function
def test_pose_detection():
    """Test pose detection with webcam or sample image."""
    detector = PoseFeatureExtractor()
    
    # Test with webcam (comment out if no camera available)
    # cap = cv2.VideoCapture(0)
    
    # For testing without webcam, create a dummy image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    pose_data = detector.detect_pose(dummy_image)
    print(f"Pose detected: {pose_data['pose_detected']}")
    
    if pose_data['pose_detected']:
        features = detector.get_pose_features(pose_data['keypoints'], pose_data['angles'])
        print(f"Feature vector shape: {features.shape}")
        print(f"Number of features: {len(features)}")
    
    return detector


if __name__ == "__main__":
    # Test the pose detection
    detector = test_pose_detection()
    print("✓ Pose detection utility initialized successfully!")
