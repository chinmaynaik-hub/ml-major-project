"""
Advanced Pose Accuracy and Validation System
Detects invalid postures and suggests similar poses with corrective feedback
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PoseValidation:
    """Validation result for a pose attempt"""
    is_valid: bool
    accuracy_score: float
    pose_name: Optional[str]
    sanskrit_name: Optional[str]
    issues: List[str]
    corrections: List[str]
    similar_poses: List[Dict]
    confidence: float

class YogaPoseValidator:
    """
    Commercial-grade pose validation system that:
    1. Analyzes pose accuracy against known templates
    2. Identifies invalid postures
    3. Suggests corrections and similar poses
    4. Provides detailed feedback for improvement
    """
    
    def __init__(self):
        self.pose_templates = self._initialize_pose_templates()
        self.min_accuracy_threshold = 0.65  # Minimum accuracy for valid pose
        self.similarity_threshold = 0.4     # Minimum similarity to suggest pose
        self.angle_tolerance = 20.0         # Degrees tolerance for angles
        self.ratio_tolerance = 0.25         # Tolerance for body ratios
        
        logger.info(f"Initialized pose validator with {len(self.pose_templates)} templates")
    
    def validate_pose(self, landmarks: np.ndarray) -> PoseValidation:
        """
        Validate the current pose and provide detailed feedback.
        
        Args:
            landmarks: MediaPipe pose landmarks array
            
        Returns:
            PoseValidation object with detailed analysis
        """
        if landmarks is None or len(landmarks) < 99:  # 33 landmarks × 3 coordinates
            return self._create_no_person_validation()
        
        try:
            # Extract pose features
            features = self._extract_pose_features(landmarks)
            if features is None:
                return self._create_invalid_pose_validation("Unable to analyze pose features")
            
            # Analyze against all templates
            pose_matches = []
            for template in self.pose_templates:
                match_result = self._analyze_pose_match(features, template)
                pose_matches.append(match_result)
            
            # Sort by accuracy score
            pose_matches.sort(key=lambda x: x['accuracy'], reverse=True)
            best_match = pose_matches[0]
            
            # Determine if pose is valid
            if best_match['accuracy'] >= self.min_accuracy_threshold:
                return self._create_valid_pose_validation(best_match, pose_matches[:3])
            else:
                return self._create_invalid_pose_validation_with_suggestions(best_match, pose_matches[:5])
                
        except Exception as e:
            logger.error(f"Pose validation error: {e}")
            return self._create_error_validation(str(e))
    
    def _extract_pose_features(self, landmarks: np.ndarray) -> Optional[Dict]:
        """Extract comprehensive pose features for analysis."""
        try:
            # Reshape landmarks if needed
            if len(landmarks.shape) == 1:
                landmarks = landmarks.reshape(33, 3)
            
            # Extract key measurements
            features = {
                'angles': self._calculate_all_angles(landmarks),
                'distances': self._calculate_key_distances(landmarks),
                'ratios': self._calculate_body_ratios(landmarks),
                'alignment': self._calculate_alignment_metrics(landmarks),
                'balance': self._calculate_balance_metrics(landmarks),
                'symmetry': self._calculate_symmetry_metrics(landmarks)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def _calculate_all_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate all relevant joint angles."""
        angles = {}
        
        # Define angle calculations with landmark indices
        angle_definitions = {
            # Arms
            'left_shoulder_angle': (13, 11, 23),    # elbow-shoulder-hip
            'right_shoulder_angle': (14, 12, 24),   # elbow-shoulder-hip
            'left_elbow_angle': (11, 13, 15),       # shoulder-elbow-wrist
            'right_elbow_angle': (12, 14, 16),      # shoulder-elbow-wrist
            
            # Legs
            'left_hip_angle': (11, 23, 25),         # shoulder-hip-knee
            'right_hip_angle': (12, 24, 26),        # shoulder-hip-knee
            'left_knee_angle': (23, 25, 27),        # hip-knee-ankle
            'right_knee_angle': (24, 26, 28),       # hip-knee-ankle
            
            # Torso
            'spine_angle': (0, 11, 23),             # nose-shoulder-hip (left side)
            'torso_lean': (11, 12, 24),             # left shoulder-right shoulder-right hip
            
            # Head and neck
            'neck_angle': (0, 11, 12),              # nose-left shoulder-right shoulder
            
            # Hip alignment
            'pelvis_angle': (23, 24, 26),           # left hip-right hip-right knee
        }
        
        for angle_name, (p1_idx, p2_idx, p3_idx) in angle_definitions.items():
            try:
                if all(idx < len(landmarks) for idx in [p1_idx, p2_idx, p3_idx]):
                    p1 = landmarks[p1_idx][:2]
                    p2 = landmarks[p2_idx][:2]  # vertex
                    p3 = landmarks[p3_idx][:2]
                    
                    angle = self._calculate_angle_3_points(p1, p2, p3)
                    angles[angle_name] = angle
            except Exception as e:
                angles[angle_name] = 0.0
                logger.debug(f"Error calculating {angle_name}: {e}")
        
        return angles
    
    def _calculate_key_distances(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate key distances between body parts."""
        distances = {}
        
        try:
            # Shoulder width
            distances['shoulder_width'] = self._distance_2d(landmarks[11][:2], landmarks[12][:2])
            
            # Hip width
            distances['hip_width'] = self._distance_2d(landmarks[23][:2], landmarks[24][:2])
            
            # Torso height
            shoulder_center = (landmarks[11][:2] + landmarks[12][:2]) / 2
            hip_center = (landmarks[23][:2] + landmarks[24][:2]) / 2
            distances['torso_height'] = self._distance_2d(shoulder_center, hip_center)
            
            # Arm lengths
            distances['left_arm_length'] = self._distance_2d(landmarks[11][:2], landmarks[15][:2])
            distances['right_arm_length'] = self._distance_2d(landmarks[12][:2], landmarks[16][:2])
            
            # Leg lengths
            distances['left_leg_length'] = self._distance_2d(landmarks[23][:2], landmarks[27][:2])
            distances['right_leg_length'] = self._distance_2d(landmarks[24][:2], landmarks[28][:2])
            
            # Feet distance
            distances['feet_distance'] = self._distance_2d(landmarks[27][:2], landmarks[28][:2])
            
        except Exception as e:
            logger.error(f"Distance calculation error: {e}")
        
        return distances
    
    def _analyze_pose_match(self, features: Dict, template: Dict) -> Dict:
        """Analyze how well current pose matches a template."""
        try:
            scores = {}
            issues = []
            corrections = []
            
            # Angle matching
            angle_score, angle_issues = self._compare_angles(features['angles'], template['angles'])
            scores['angles'] = angle_score
            issues.extend(angle_issues)
            
            # Ratio matching
            ratio_score, ratio_issues = self._compare_ratios(features['ratios'], template['ratios'])
            scores['ratios'] = ratio_score
            issues.extend(ratio_issues)
            
            # Alignment matching
            alignment_score = self._compare_alignment(features['alignment'], template.get('alignment', {}))
            scores['alignment'] = alignment_score
            
            # Balance matching
            balance_score = self._compare_balance(features['balance'], template.get('balance', {}))
            scores['balance'] = balance_score
            
            # Calculate overall accuracy
            weights = {'angles': 0.4, 'ratios': 0.25, 'alignment': 0.2, 'balance': 0.15}
            accuracy = sum(scores[key] * weights[key] for key in scores.keys())
            
            # Generate corrections based on issues
            corrections = self._generate_corrections(template['name'], issues, features)
            
            return {
                'template': template,
                'accuracy': accuracy,
                'component_scores': scores,
                'issues': issues,
                'corrections': corrections,
                'pose_name': template['name'],
                'sanskrit_name': template['sanskrit_name'],
                'description': template['description']
            }
            
        except Exception as e:
            logger.error(f"Pose matching error: {e}")
            return {
                'template': template,
                'accuracy': 0.0,
                'component_scores': {},
                'issues': [f"Analysis error: {str(e)}"],
                'corrections': [],
                'pose_name': template['name'],
                'sanskrit_name': template['sanskrit_name']
            }
    
    def _compare_angles(self, current_angles: Dict, template_angles: Dict) -> Tuple[float, List[str]]:
        """Compare current angles with template angles."""
        if not current_angles or not template_angles:
            return 0.0, ["Unable to analyze joint angles"]
        
        angle_scores = []
        issues = []
        
        for angle_name, target_angle in template_angles.items():
            if angle_name in current_angles:
                current_angle = current_angles[angle_name]
                angle_diff = abs(current_angle - target_angle)
                
                # Calculate score (0-1 scale)
                score = max(0.0, 1.0 - (angle_diff / self.angle_tolerance))
                angle_scores.append(score)
                
                # Identify issues
                if angle_diff > self.angle_tolerance:
                    if current_angle > target_angle:
                        issues.append(f"{angle_name.replace('_', ' ').title()}: too wide by {angle_diff:.1f}°")
                    else:
                        issues.append(f"{angle_name.replace('_', ' ').title()}: too narrow by {angle_diff:.1f}°")
            else:
                angle_scores.append(0.0)
                issues.append(f"Cannot measure {angle_name.replace('_', ' ')}")
        
        return np.mean(angle_scores) if angle_scores else 0.0, issues
    
    def _compare_ratios(self, current_ratios: Dict, template_ratios: Dict) -> Tuple[float, List[str]]:
        """Compare body ratios with template."""
        if not current_ratios or not template_ratios:
            return 0.0, ["Unable to analyze body proportions"]
        
        ratio_scores = []
        issues = []
        
        for ratio_name, target_ratio in template_ratios.items():
            if ratio_name in current_ratios and target_ratio > 0:
                current_ratio = current_ratios[ratio_name]
                ratio_diff = abs(current_ratio - target_ratio) / target_ratio
                
                score = max(0.0, 1.0 - (ratio_diff / self.ratio_tolerance))
                ratio_scores.append(score)
                
                if ratio_diff > self.ratio_tolerance:
                    issues.append(f"Incorrect {ratio_name.replace('_', ' ')} proportion")
            else:
                ratio_scores.append(0.0)
        
        return np.mean(ratio_scores) if ratio_scores else 0.0, issues
    
    def _generate_corrections(self, pose_name: str, issues: List[str], features: Dict) -> List[str]:
        """Generate specific corrections based on pose and issues."""
        corrections = []
        
        # Pose-specific corrections
        pose_corrections = {
            'tadasana': [
                "Stand with feet hip-width apart",
                "Keep arms relaxed at your sides",
                "Engage your core and lift your chest",
                "Keep your head in neutral position"
            ],
            'vriksasana': [
                "Place your foot on inner thigh or calf (not on knee)",
                "Keep your standing leg strong and straight",
                "Bring palms together at heart center or raise arms overhead",
                "Find a focal point to maintain balance"
            ],
            'uttanasana': [
                "Hinge forward from your hips, not your waist",
                "Keep a slight bend in your knees",
                "Let your arms hang naturally or hold opposite elbows",
                "Keep your weight evenly distributed on both feet"
            ],
            'bhujangasana': [
                "Keep your pelvis grounded",
                "Press through your palms to lift your chest",
                "Keep your shoulders away from your ears",
                "Engage your back muscles, don't rely only on arms"
            ],
            'adho mukha svanasana': [
                "Create an inverted V shape with your body",
                "Keep your hands shoulder-width apart",
                "Straighten your legs and push your heels toward the ground",
                "Keep your head between your arms"
            ]
        }
        
        if pose_name in pose_corrections:
            corrections.extend(pose_corrections[pose_name])
        
        # Add issue-specific corrections
        for issue in issues:
            if "elbow" in issue.lower():
                corrections.append("Adjust your arm position and elbow angle")
            elif "knee" in issue.lower():
                corrections.append("Check your leg alignment and knee position")
            elif "hip" in issue.lower():
                corrections.append("Focus on proper hip alignment and engagement")
            elif "spine" in issue.lower():
                corrections.append("Work on spinal alignment and posture")
        
        return list(set(corrections))  # Remove duplicates
    
    def _initialize_pose_templates(self) -> List[Dict]:
        """Initialize pose templates with detailed requirements."""
        templates = [
            {
                'name': 'tadasana',
                'sanskrit_name': 'ताड़ासन (Tadasana)',
                'description': 'Mountain Pose - standing tall and straight',
                'difficulty': 1,
                'angles': {
                    'left_elbow_angle': 180.0,
                    'right_elbow_angle': 180.0,
                    'left_knee_angle': 180.0,
                    'right_knee_angle': 180.0,
                    'spine_angle': 90.0
                },
                'ratios': {
                    'shoulder_to_hip': 0.9,
                    'feet_distance': 0.3,
                    'arm_symmetry': 0.95
                },
                'alignment': {
                    'spine_straight': True,
                    'shoulders_level': True
                },
                'balance': {
                    'weight_distribution': 'even'
                }
            },
            {
                'name': 'vriksasana',
                'sanskrit_name': 'वृक्षासन (Vrikshasana)',
                'description': 'Tree Pose - single leg balance',
                'difficulty': 3,
                'angles': {
                    'left_knee_angle': 90.0,      # Bent leg
                    'right_knee_angle': 180.0,    # Standing leg
                    'left_hip_angle': 45.0,       # Hip opening
                    'spine_angle': 90.0
                },
                'ratios': {
                    'single_leg_support': True,
                    'arm_elevation': 0.8
                },
                'alignment': {
                    'spine_straight': True,
                    'single_leg_balance': True
                },
                'balance': {
                    'weight_distribution': 'single_leg'
                }
            },
            {
                'name': 'uttanasana',
                'sanskrit_name': 'उत्तानासन (Uttanasana)',
                'description': 'Standing Forward Fold',
                'difficulty': 2,
                'angles': {
                    'left_hip_angle': 45.0,
                    'right_hip_angle': 45.0,
                    'left_knee_angle': 170.0,     # Slight bend allowed
                    'right_knee_angle': 170.0,
                    'spine_angle': 45.0           # Forward fold
                },
                'ratios': {
                    'forward_fold_depth': 0.7,
                    'feet_distance': 0.3
                },
                'alignment': {
                    'forward_fold': True,
                    'even_weight': True
                }
            },
            {
                'name': 'bhujangasana',
                'sanskrit_name': 'भुजंगासन (Bhujangasana)',
                'description': 'Cobra Pose - gentle backbend',
                'difficulty': 3,
                'angles': {
                    'left_elbow_angle': 120.0,
                    'right_elbow_angle': 120.0,
                    'spine_angle': 120.0,         # Backbend
                    'left_hip_angle': 160.0,
                    'right_hip_angle': 160.0
                },
                'ratios': {
                    'chest_lift': 0.6,
                    'arm_support': 0.4
                },
                'alignment': {
                    'chest_lifted': True,
                    'pelvis_grounded': True
                }
            },
            {
                'name': 'adho mukha svanasana',
                'sanskrit_name': 'अधो मुख श्वानासन (Adho Mukha Svanasana)',
                'description': 'Downward Facing Dog',
                'difficulty': 2,
                'angles': {
                    'left_elbow_angle': 180.0,
                    'right_elbow_angle': 180.0,
                    'left_knee_angle': 180.0,
                    'right_knee_angle': 180.0,
                    'left_hip_angle': 90.0,
                    'right_hip_angle': 90.0
                },
                'ratios': {
                    'inverted_v': True,
                    'hand_foot_distance': 1.2
                },
                'alignment': {
                    'inverted_v_shape': True,
                    'weight_distributed': True
                }
            }
        ]
        
        return templates
    
    def _create_valid_pose_validation(self, best_match: Dict, alternatives: List[Dict]) -> PoseValidation:
        """Create validation result for valid pose."""
        return PoseValidation(
            is_valid=True,
            accuracy_score=best_match['accuracy'],
            pose_name=best_match['pose_name'],
            sanskrit_name=best_match['sanskrit_name'],
            issues=best_match['issues'][:2] if best_match['issues'] else [],  # Show only top 2 issues
            corrections=best_match['corrections'][:3],  # Show top 3 corrections
            similar_poses=[],  # No alternatives needed for valid pose
            confidence=min(best_match['accuracy'] * 1.2, 1.0)
        )
    
    def _create_invalid_pose_validation_with_suggestions(self, best_match: Dict, all_matches: List[Dict]) -> PoseValidation:
        """Create validation result for invalid pose with suggestions."""
        # Find poses with reasonable similarity for suggestions
        similar_poses = []
        for match in all_matches:
            if match['accuracy'] >= self.similarity_threshold:
                similar_poses.append({
                    'pose_name': match['pose_name'],
                    'sanskrit_name': match['sanskrit_name'],
                    'description': match['description'],
                    'similarity': match['accuracy'],
                    'difficulty': match['template'].get('difficulty', 3)
                })
        
        # Sort by similarity and difficulty (easier poses first if similar accuracy)
        similar_poses.sort(key=lambda x: (-x['similarity'], x['difficulty']))
        
        return PoseValidation(
            is_valid=False,
            accuracy_score=best_match['accuracy'],
            pose_name=None,
            sanskrit_name=None,
            issues=[
                "Current posture is not a recognized yoga pose",
                f"Closest match: {best_match['pose_name']} ({best_match['accuracy']*100:.1f}% similarity)",
                "Position may be unsafe or incorrect"
            ],
            corrections=[
                "Try one of the suggested poses below",
                "Ensure you are fully visible in the camera",
                "Check your pose alignment and form",
                "Consider starting with easier poses if you're a beginner"
            ],
            similar_poses=similar_poses[:4],  # Show top 4 suggestions
            confidence=0.0
        )
    
    def _create_invalid_pose_validation(self, reason: str) -> PoseValidation:
        """Create validation result for invalid pose without suggestions."""
        return PoseValidation(
            is_valid=False,
            accuracy_score=0.0,
            pose_name=None,
            sanskrit_name=None,
            issues=[reason],
            corrections=[
                "Ensure you are fully visible in the camera",
                "Try a basic pose like Mountain Pose (Tadasana)",
                "Check your lighting and camera positioning"
            ],
            similar_poses=[
                {
                    'pose_name': 'tadasana',
                    'sanskrit_name': 'ताड़ासन (Tadasana)',
                    'description': 'Mountain Pose - good starting pose',
                    'similarity': 0.0,
                    'difficulty': 1
                }
            ],
            confidence=0.0
        )
    
    def _create_no_person_validation(self) -> PoseValidation:
        """Create validation result when no person is detected."""
        return PoseValidation(
            is_valid=False,
            accuracy_score=0.0,
            pose_name=None,
            sanskrit_name=None,
            issues=["No person detected in the camera"],
            corrections=[
                "Step into the camera view",
                "Ensure good lighting",
                "Make sure you are fully visible from head to toe",
                "Check camera permissions"
            ],
            similar_poses=[],
            confidence=0.0
        )
    
    def _create_error_validation(self, error_msg: str) -> PoseValidation:
        """Create validation result for system errors."""
        return PoseValidation(
            is_valid=False,
            accuracy_score=0.0,
            pose_name=None,
            sanskrit_name=None,
            issues=[f"System error: {error_msg}"],
            corrections=["Please try again", "Check your camera connection"],
            similar_poses=[],
            confidence=0.0
        )
    
    # Helper methods
    def _distance_2d(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate 2D Euclidean distance."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _calculate_angle_3_points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points (p2 is vertex)."""
        try:
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate dot product and magnitudes
            dot_product = np.dot(v1, v2)
            magnitude1 = np.linalg.norm(v1)
            magnitude2 = np.linalg.norm(v2)
            
            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            # Calculate cosine of angle
            cos_angle = dot_product / (magnitude1 * magnitude2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to valid range
            
            # Convert to degrees
            angle = np.arccos(cos_angle) * 180 / np.pi
            return angle
            
        except Exception:
            return 0.0
    
    def _calculate_body_ratios(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate body proportion ratios."""
        ratios = {}
        try:
            # Basic measurements
            shoulder_width = self._distance_2d(landmarks[11][:2], landmarks[12][:2])
            hip_width = self._distance_2d(landmarks[23][:2], landmarks[24][:2])
            torso_height = self._distance_2d(
                (landmarks[11][:2] + landmarks[12][:2]) / 2,
                (landmarks[23][:2] + landmarks[24][:2]) / 2
            )
            
            # Calculate ratios safely
            if shoulder_width > 0:
                ratios['shoulder_to_hip'] = hip_width / shoulder_width
            if torso_height > 0:
                ratios['shoulder_to_torso'] = shoulder_width / torso_height
            
            # Add more ratios as needed
            ratios['feet_distance'] = self._distance_2d(landmarks[27][:2], landmarks[28][:2]) / max(torso_height, 0.1)
            
        except Exception as e:
            logger.error(f"Ratio calculation error: {e}")
        
        return ratios
    
    def _calculate_alignment_metrics(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate pose alignment metrics."""
        alignment = {}
        try:
            # Shoulder alignment
            left_shoulder = landmarks[11][:2]
            right_shoulder = landmarks[12][:2]
            shoulder_level_diff = abs(left_shoulder[1] - right_shoulder[1])
            alignment['shoulder_level'] = 1.0 - min(shoulder_level_diff / 0.1, 1.0)
            
            # Hip alignment
            left_hip = landmarks[23][:2]
            right_hip = landmarks[24][:2]
            hip_level_diff = abs(left_hip[1] - right_hip[1])
            alignment['hip_level'] = 1.0 - min(hip_level_diff / 0.1, 1.0)
            
        except Exception:
            alignment['shoulder_level'] = 0.0
            alignment['hip_level'] = 0.0
        
        return alignment
    
    def _calculate_balance_metrics(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate balance and stability metrics."""
        balance = {}
        try:
            # Center of mass calculation
            key_points = [landmarks[11], landmarks[12], landmarks[23], landmarks[24]]
            center_of_mass = np.mean(key_points, axis=0)[:2]
            
            # Base of support
            left_foot = landmarks[27][:2]
            right_foot = landmarks[28][:2]
            support_center = (left_foot + right_foot) / 2
            
            # Balance metrics
            balance['stability'] = 1.0 - min(self._distance_2d(center_of_mass, support_center) / 0.2, 1.0)
            balance['support_width'] = self._distance_2d(left_foot, right_foot)
            
        except Exception:
            balance['stability'] = 0.0
            balance['support_width'] = 0.0
        
        return balance
    
    def _calculate_symmetry_metrics(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate body symmetry metrics."""
        symmetry = {}
        try:
            # Arm symmetry
            left_arm = self._distance_2d(landmarks[11][:2], landmarks[15][:2])
            right_arm = self._distance_2d(landmarks[12][:2], landmarks[16][:2])
            if left_arm > 0 and right_arm > 0:
                symmetry['arm_symmetry'] = min(left_arm / right_arm, right_arm / left_arm)
            
            # Leg symmetry
            left_leg = self._distance_2d(landmarks[23][:2], landmarks[27][:2])
            right_leg = self._distance_2d(landmarks[24][:2], landmarks[28][:2])
            if left_leg > 0 and right_leg > 0:
                symmetry['leg_symmetry'] = min(left_leg / right_leg, right_leg / left_leg)
            
        except Exception:
            symmetry['arm_symmetry'] = 0.0
            symmetry['leg_symmetry'] = 0.0
        
        return symmetry
    
    def _compare_alignment(self, current_alignment: Dict, template_alignment: Dict) -> float:
        """Compare current alignment with template requirements."""
        if not template_alignment:
            return 0.8  # Default score if no alignment requirements
        
        scores = []
        for key, required in template_alignment.items():
            if key in current_alignment:
                if isinstance(required, bool):
                    # Boolean alignment requirements
                    score = 1.0 if current_alignment[key] > 0.7 else 0.0
                else:
                    # Numeric alignment requirements
                    score = min(current_alignment[key], 1.0)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.8
    
    def _compare_balance(self, current_balance: Dict, template_balance: Dict) -> float:
        """Compare current balance with template requirements."""
        if not template_balance:
            return 0.8  # Default score if no balance requirements
        
        # Simple balance scoring based on stability
        stability_score = current_balance.get('stability', 0.5)
        return stability_score
