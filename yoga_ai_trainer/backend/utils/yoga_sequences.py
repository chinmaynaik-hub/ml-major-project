"""
Yoga Sequence System for Yoga AI Trainer
Implements yoga flows like Surya Namaskara with progression tracking and timing
"""

import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SequenceState(Enum):
    """States for sequence tracking."""
    READY = "ready"
    IN_PROGRESS = "in_progress" 
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class PoseStep:
    """A single pose step in a yoga sequence."""
    pose_name: str
    sanskrit_name: str
    pronunciation: str
    hold_time: int  # seconds
    transition_time: int  # seconds to transition
    instructions: List[str]
    breath_count: Optional[int] = None
    difficulty: int = 2  # 1-5 scale
    
    def to_dict(self):
        return asdict(self)

@dataclass
class SequenceProgress:
    """Progress tracking for a yoga sequence."""
    current_step: int
    total_steps: int
    elapsed_time: float
    poses_completed: List[str]
    current_pose_start_time: float
    sequence_start_time: float
    accuracy_scores: List[float]
    
    def completion_percentage(self) -> float:
        return (self.current_step / self.total_steps) * 100

class YogaSequence:
    """
    A complete yoga sequence with poses, timing, and guidance.
    """
    
    def __init__(self, name: str, description: str, poses: List[PoseStep], 
                 difficulty: int = 2, duration_minutes: int = 10):
        self.name = name
        self.description = description
        self.poses = poses
        self.difficulty = difficulty
        self.duration_minutes = duration_minutes
        self.total_poses = len(poses)
        
        # Calculate total estimated time
        self.estimated_duration = sum(pose.hold_time + pose.transition_time for pose in poses)
        
    def get_pose_at_index(self, index: int) -> Optional[PoseStep]:
        """Get pose at specific index."""
        if 0 <= index < len(self.poses):
            return self.poses[index]
        return None
    
    def to_dict(self) -> Dict:
        """Convert sequence to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'difficulty': self.difficulty,
            'duration_minutes': self.duration_minutes,
            'total_poses': self.total_poses,
            'estimated_duration': self.estimated_duration,
            'poses': [pose.to_dict() for pose in self.poses]
        }

class YogaSequenceTracker:
    """
    Tracks progress through yoga sequences with timing and pose validation.
    """
    
    def __init__(self):
        self.current_sequence: Optional[YogaSequence] = None
        self.progress: Optional[SequenceProgress] = None
        self.state = SequenceState.READY
        self.sequences = self._load_built_in_sequences()
        
        logger.info("Yoga sequence tracker initialized")
    
    def start_sequence(self, sequence_name: str) -> bool:
        """
        Start a yoga sequence.
        
        Args:
            sequence_name: Name of the sequence to start
            
        Returns:
            True if sequence started successfully
        """
        if sequence_name not in self.sequences:
            logger.error(f"Sequence '{sequence_name}' not found")
            return False
        
        self.current_sequence = self.sequences[sequence_name]
        
        # Initialize progress tracking
        current_time = time.time()
        self.progress = SequenceProgress(
            current_step=0,
            total_steps=self.current_sequence.total_poses,
            elapsed_time=0.0,
            poses_completed=[],
            current_pose_start_time=current_time,
            sequence_start_time=current_time,
            accuracy_scores=[]
        )
        
        self.state = SequenceState.IN_PROGRESS
        
        logger.info(f"Started sequence: {sequence_name}")
        return True
    
    def get_current_pose(self) -> Optional[PoseStep]:
        """Get the current pose in the sequence."""
        if not self.current_sequence or not self.progress:
            return None
        
        return self.current_sequence.get_pose_at_index(self.progress.current_step)
    
    def advance_to_next_pose(self, accuracy_score: float = 0.0) -> bool:
        """
        Move to the next pose in the sequence.
        
        Args:
            accuracy_score: Accuracy score for the completed pose (0.0-1.0)
            
        Returns:
            True if advanced successfully, False if sequence is complete
        """
        if not self.current_sequence or not self.progress:
            return False
        
        current_pose = self.get_current_pose()
        if current_pose:
            # Record completion of current pose
            self.progress.poses_completed.append(current_pose.pose_name)
            self.progress.accuracy_scores.append(accuracy_score)
        
        # Move to next pose
        self.progress.current_step += 1
        
        # Update timing
        current_time = time.time()
        self.progress.elapsed_time = current_time - self.progress.sequence_start_time
        self.progress.current_pose_start_time = current_time
        
        # Check if sequence is complete
        if self.progress.current_step >= self.current_sequence.total_poses:
            self.state = SequenceState.COMPLETED
            logger.info(f"Sequence completed: {self.current_sequence.name}")
            return False
        
        logger.info(f"Advanced to pose {self.progress.current_step + 1}/{self.progress.total_steps}")
        return True
    
    def pause_sequence(self):
        """Pause the current sequence."""
        if self.state == SequenceState.IN_PROGRESS:
            self.state = SequenceState.PAUSED
            logger.info("Sequence paused")
    
    def resume_sequence(self):
        """Resume the paused sequence."""
        if self.state == SequenceState.PAUSED:
            self.state = SequenceState.IN_PROGRESS
            # Update timing
            if self.progress:
                self.progress.current_pose_start_time = time.time()
            logger.info("Sequence resumed")
    
    def cancel_sequence(self):
        """Cancel the current sequence."""
        self.state = SequenceState.CANCELLED
        logger.info("Sequence cancelled")
    
    def get_progress_info(self) -> Optional[Dict]:
        """Get detailed progress information."""
        if not self.current_sequence or not self.progress:
            return None
        
        current_pose = self.get_current_pose()
        
        # Calculate time remaining in current pose
        time_in_current_pose = time.time() - self.progress.current_pose_start_time
        time_remaining = max(0, current_pose.hold_time - time_in_current_pose) if current_pose else 0
        
        return {
            'sequence_name': self.current_sequence.name,
            'state': self.state.value,
            'progress': {
                'current_step': self.progress.current_step + 1,  # 1-indexed for display
                'total_steps': self.progress.total_steps,
                'completion_percentage': self.progress.completion_percentage(),
                'elapsed_time': self.progress.elapsed_time,
                'poses_completed': self.progress.poses_completed,
                'average_accuracy': sum(self.progress.accuracy_scores) / len(self.progress.accuracy_scores) if self.progress.accuracy_scores else 0.0
            },
            'current_pose': current_pose.to_dict() if current_pose else None,
            'timing': {
                'time_in_current_pose': time_in_current_pose,
                'time_remaining_in_pose': time_remaining,
                'estimated_total_duration': self.current_sequence.estimated_duration,
                'estimated_time_remaining': self.current_sequence.estimated_duration - self.progress.elapsed_time
            },
            'next_pose': self.current_sequence.get_pose_at_index(self.progress.current_step + 1).to_dict() if self.progress.current_step + 1 < len(self.current_sequence.poses) else None
        }
    
    def get_available_sequences(self) -> List[Dict]:
        """Get list of available yoga sequences."""
        return [
            {
                'name': name,
                'description': seq.description,
                'difficulty': seq.difficulty,
                'duration_minutes': seq.duration_minutes,
                'total_poses': seq.total_poses,
                'estimated_duration': seq.estimated_duration
            }
            for name, seq in self.sequences.items()
        ]
    
    def _load_built_in_sequences(self) -> Dict[str, YogaSequence]:
        """Load built-in yoga sequences."""
        sequences = {}
        
        # Surya Namaskara A (Sun Salutation A)
        surya_namaskara_a_poses = [
            PoseStep(
                pose_name="tadasana",
                sanskrit_name="ताड़ासन",
                pronunciation="ta-DAH-sa-na",
                hold_time=8,
                transition_time=2,
                instructions=[
                    "Stand tall with feet together",
                    "Hands at your sides, shoulders relaxed",
                    "Engage your core and breathe deeply"
                ],
                breath_count=2
            ),
            PoseStep(
                pose_name="urdhva_hastasana",
                sanskrit_name="ऊर्ध्व हस्तासन",
                pronunciation="OORD-vah has-TAH-sa-na",
                hold_time=6,
                transition_time=3,
                instructions=[
                    "Inhale, sweep arms overhead",
                    "Palms face each other or touch",
                    "Lift through the crown of your head"
                ],
                breath_count=1
            ),
            PoseStep(
                pose_name="uttanasana",
                sanskrit_name="उत्तानासन", 
                pronunciation="oot-tan-AH-sa-na",
                hold_time=10,
                transition_time=3,
                instructions=[
                    "Exhale, hinge forward from hips",
                    "Keep slight bend in knees",
                    "Let arms hang or hold opposite elbows"
                ],
                breath_count=2
            ),
            PoseStep(
                pose_name="ardha_uttanasana",
                sanskrit_name="अर्ध उत्तानासन",
                pronunciation="AR-da oot-tan-AH-sa-na", 
                hold_time=6,
                transition_time=2,
                instructions=[
                    "Inhale, hands to shins or fingertips to floor",
                    "Lengthen spine, open heart",
                    "Gaze forward, shoulders over wrists"
                ],
                breath_count=1
            ),
            PoseStep(
                pose_name="chaturanga_dandasana",
                sanskrit_name="चतुरंग दंडासन",
                pronunciation="chat-tour-AN-ga dan-DAH-sa-na",
                hold_time=8,
                transition_time=3,
                instructions=[
                    "Step or jump back to plank",
                    "Lower down keeping elbows close to body",
                    "Keep core engaged, body in one line"
                ],
                breath_count=1,
                difficulty=3
            ),
            PoseStep(
                pose_name="urdhva_mukha_svanasana",
                sanskrit_name="ऊर्ध्व मुख श्वानासन",
                pronunciation="OORD-vah MU-kha shva-NAH-sa-na",
                hold_time=8,
                transition_time=3,
                instructions=[
                    "Roll over toes, straighten arms",
                    "Lift chest and legs off ground",
                    "Press hands down, open heart"
                ],
                breath_count=1,
                difficulty=3
            ),
            PoseStep(
                pose_name="adho_mukha_svanasana",
                sanskrit_name="अधो मुख श्वानासन",
                pronunciation="AH-do MU-kha shva-NAH-sa-na", 
                hold_time=15,
                transition_time=2,
                instructions=[
                    "Roll over toes to downward dog",
                    "Press hands down, lift hips up and back",
                    "Straighten legs, ground through feet"
                ],
                breath_count=3
            ),
            PoseStep(
                pose_name="uttanasana",
                sanskrit_name="उत्तानासन",
                pronunciation="oot-tan-AH-sa-na",
                hold_time=8,
                transition_time=3,
                instructions=[
                    "Step or jump feet to hands",
                    "Fold forward from hips",
                    "Soften knees if needed"
                ],
                breath_count=1
            ),
            PoseStep(
                pose_name="tadasana",
                sanskrit_name="ताड़ासन",
                pronunciation="ta-DAH-sa-na",
                hold_time=10,
                transition_time=0,
                instructions=[
                    "Inhale, rise to standing",
                    "Hands at sides or overhead",
                    "Feel the energy of the sequence"
                ],
                breath_count=2
            )
        ]
        
        sequences["surya_namaskara_a"] = YogaSequence(
            name="Surya Namaskara A",
            description="Traditional Sun Salutation A - energizing morning flow",
            poses=surya_namaskara_a_poses,
            difficulty=2,
            duration_minutes=8
        )
        
        # Basic Standing Sequence
        standing_sequence_poses = [
            PoseStep(
                pose_name="tadasana",
                sanskrit_name="ताड़ासन",
                pronunciation="ta-DAH-sa-na",
                hold_time=15,
                transition_time=3,
                instructions=[
                    "Stand with feet hip-width apart",
                    "Ground through all four corners of feet",
                    "Lengthen spine, crown of head reaching up"
                ],
                breath_count=3
            ),
            PoseStep(
                pose_name="vriksasana",
                sanskrit_name="वृक्षासन",
                pronunciation="vrik-SHAH-sa-na",
                hold_time=20,
                transition_time=5,
                instructions=[
                    "Place right foot on inner left thigh",
                    "Press foot into leg, leg into foot",
                    "Hands at heart center or overhead"
                ],
                breath_count=4
            ),
            PoseStep(
                pose_name="vriksasana",
                sanskrit_name="वृक्षासन",
                pronunciation="vrik-SHAH-sa-na",
                hold_time=20,
                transition_time=3,
                instructions=[
                    "Switch sides - left foot on right thigh",
                    "Find your drishti (focused gaze)",
                    "Breathe steadily and deeply"
                ],
                breath_count=4
            ),
            PoseStep(
                pose_name="trikonasana",
                sanskrit_name="त्रिकोणासन",
                pronunciation="trik-cone-AH-sa-na",
                hold_time=25,
                transition_time=5,
                instructions=[
                    "Wide-legged stance, right toes forward",
                    "Reach right hand toward floor or shin",
                    "Left arm reaches toward ceiling"
                ],
                breath_count=5
            ),
            PoseStep(
                pose_name="trikonasana",
                sanskrit_name="त्रिकोणासन", 
                pronunciation="trik-cone-AH-sa-na",
                hold_time=25,
                transition_time=3,
                instructions=[
                    "Switch sides - left toes forward",
                    "Keep both legs straight and strong",
                    "Breathe into the side body stretch"
                ],
                breath_count=5
            ),
            PoseStep(
                pose_name="tadasana",
                sanskrit_name="ताड़ासन",
                pronunciation="ta-DAH-sa-na",
                hold_time=15,
                transition_time=0,
                instructions=[
                    "Return to mountain pose",
                    "Feel the effects of your practice",
                    "Take a moment of gratitude"
                ],
                breath_count=3
            )
        ]
        
        sequences["standing_basics"] = YogaSequence(
            name="Standing Basics",
            description="Fundamental standing poses for strength and balance",
            poses=standing_sequence_poses,
            difficulty=1,
            duration_minutes=12
        )
        
        # Restorative Sequence
        restorative_poses = [
            PoseStep(
                pose_name="balasana",
                sanskrit_name="बालासन",
                pronunciation="bah-LAH-sa-na",
                hold_time=45,
                transition_time=5,
                instructions=[
                    "Kneel with big toes touching",
                    "Sit back on heels, fold forward",
                    "Arms extended or by your sides"
                ],
                breath_count=8,
                difficulty=1
            ),
            PoseStep(
                pose_name="bhujangasana",
                sanskrit_name="भुजंगासन",
                pronunciation="bhu-jan-GAH-sa-na",
                hold_time=30,
                transition_time=5,
                instructions=[
                    "Lie on belly, palms under shoulders",
                    "Press palms to lift chest",
                    "Keep pelvis grounded"
                ],
                breath_count=6
            ),
            PoseStep(
                pose_name="balasana",
                sanskrit_name="बालासन",
                pronunciation="bah-LAH-sa-na",
                hold_time=30,
                transition_time=5,
                instructions=[
                    "Return to child's pose",
                    "Counter the backbend",
                    "Rest and breathe deeply"
                ],
                breath_count=6
            ),
            PoseStep(
                pose_name="shavasana",
                sanskrit_name="शवासन",
                pronunciation="sha-VAH-sa-na",
                hold_time=180,
                transition_time=0,
                instructions=[
                    "Lie on your back, arms by sides",
                    "Let your entire body relax",
                    "Focus on natural breath"
                ],
                breath_count=20,
                difficulty=1
            )
        ]
        
        sequences["restorative"] = YogaSequence(
            name="Restorative Practice",
            description="Gentle, relaxing poses for stress relief and recovery",
            poses=restorative_poses,
            difficulty=1,
            duration_minutes=8
        )
        
        return sequences
    
    def save_custom_sequence(self, sequence: YogaSequence, filepath: str):
        """Save a custom sequence to file."""
        data = sequence.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved custom sequence to {filepath}")
    
    def load_custom_sequence(self, filepath: str) -> Optional[YogaSequence]:
        """Load a custom sequence from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            poses = [PoseStep(**pose_data) for pose_data in data['poses']]
            
            sequence = YogaSequence(
                name=data['name'],
                description=data['description'],
                poses=poses,
                difficulty=data.get('difficulty', 2),
                duration_minutes=data.get('duration_minutes', 10)
            )
            
            self.sequences[data['name']] = sequence
            logger.info(f"Loaded custom sequence: {data['name']}")
            return sequence
            
        except Exception as e:
            logger.error(f"Error loading custom sequence: {e}")
            return None

# Global sequence tracker instance
_sequence_tracker_instance = None

def get_sequence_tracker() -> YogaSequenceTracker:
    """Get or create the global sequence tracker instance."""
    global _sequence_tracker_instance
    if _sequence_tracker_instance is None:
        _sequence_tracker_instance = YogaSequenceTracker()
    return _sequence_tracker_instance
