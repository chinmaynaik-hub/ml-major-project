"""
Voice Feedback System for Yoga AI Trainer
Real-time Sanskrit pronunciation and pose guidance using text-to-speech
"""

import pyttsx3
import threading
import time
from typing import Optional, Dict, List
import logging
import json
from pathlib import Path
import queue

logger = logging.getLogger(__name__)

class VoiceFeedbackSystem:
    """
    Advanced voice feedback system for yoga pose guidance.
    
    Features:
    - Real-time Sanskrit pronunciation
    - Pose correction guidance
    - Audio cues for transitions
    - Configurable voice properties
    - Thread-safe operation
    """
    
    def __init__(self, voice_rate: int = 150, voice_volume: float = 0.8):
        """
        Initialize the voice feedback system.
        
        Args:
            voice_rate: Speech rate (words per minute)
            voice_volume: Volume level (0.0 - 1.0)
        """
        self.voice_rate = voice_rate
        self.voice_volume = voice_volume
        self.enabled = True
        
        # Voice queue for thread-safe operation
        self.voice_queue = queue.Queue()
        self.is_speaking = False
        
        # Initialize TTS engine
        self._init_tts_engine()
        
        # Start voice worker thread
        self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self.voice_thread.start()
        
        # Load voice templates
        self.templates = self._load_voice_templates()
        
        # Pose transition timing
        self.last_pose_announcement = 0
        self.min_announcement_interval = 3.0  # Seconds between announcements
        
        logger.info("Voice feedback system initialized")
    
    def _init_tts_engine(self):
        """Initialize the text-to-speech engine."""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure voice properties
            self.tts_engine.setProperty('rate', self.voice_rate)
            self.tts_engine.setProperty('volume', self.voice_volume)
            
            # Try to set a pleasant voice
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
        except Exception as e:
            logger.error(f"Error initializing TTS engine: {e}")
            self.tts_engine = None
    
    def _voice_worker(self):
        """Background worker thread for processing voice feedback."""
        while True:
            try:
                # Get next voice command from queue
                voice_command = self.voice_queue.get(timeout=1.0)
                
                if voice_command is None:  # Shutdown signal
                    break
                
                if not self.enabled or self.tts_engine is None:
                    continue
                
                self.is_speaking = True
                
                # Process different types of voice commands
                if voice_command['type'] == 'pose_announcement':
                    self._speak_pose_announcement(voice_command)
                elif voice_command['type'] == 'correction':
                    self._speak_correction(voice_command)
                elif voice_command['type'] == 'transition':
                    self._speak_transition(voice_command)
                elif voice_command['type'] == 'encouragement':
                    self._speak_encouragement(voice_command)
                elif voice_command['type'] == 'custom':
                    self._speak_text(voice_command['text'])
                
                self.is_speaking = False
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Voice worker error: {e}")
                self.is_speaking = False
    
    def announce_pose(self, pose_name: str, sanskrit_name: str = None, pronunciation: str = None):
        """
        Announce a detected yoga pose with Sanskrit pronunciation.
        
        Args:
            pose_name: English pose name
            sanskrit_name: Sanskrit name (optional)
            pronunciation: Phonetic pronunciation (optional)
        """
        current_time = time.time()
        
        # Avoid too frequent announcements
        if current_time - self.last_pose_announcement < self.min_announcement_interval:
            return
        
        self.last_pose_announcement = current_time
        
        voice_command = {
            'type': 'pose_announcement',
            'pose_name': pose_name,
            'sanskrit_name': sanskrit_name,
            'pronunciation': pronunciation
        }
        
        self.voice_queue.put(voice_command)
    
    def provide_correction(self, corrections: List[str], pose_name: str = None):
        """
        Provide voice feedback for pose corrections.
        
        Args:
            corrections: List of correction instructions
            pose_name: Current pose name for context
        """
        if not corrections:
            return
        
        voice_command = {
            'type': 'correction',
            'corrections': corrections,
            'pose_name': pose_name
        }
        
        self.voice_queue.put(voice_command)
    
    def announce_transition(self, from_pose: str, to_pose: str, hold_time: int = None):
        """
        Announce transition between poses.
        
        Args:
            from_pose: Current pose
            to_pose: Next pose
            hold_time: Time to hold current pose (seconds)
        """
        voice_command = {
            'type': 'transition',
            'from_pose': from_pose,
            'to_pose': to_pose,
            'hold_time': hold_time
        }
        
        self.voice_queue.put(voice_command)
    
    def provide_encouragement(self, context: str = 'general'):
        """
        Provide encouraging feedback.
        
        Args:
            context: Context for encouragement (general, improvement, achievement)
        """
        voice_command = {
            'type': 'encouragement',
            'context': context
        }
        
        self.voice_queue.put(voice_command)
    
    def speak_custom(self, text: str, priority: bool = False):
        """
        Speak custom text.
        
        Args:
            text: Text to speak
            priority: If True, clear queue and speak immediately
        """
        if priority and not self.voice_queue.empty():
            # Clear queue for priority messages
            while not self.voice_queue.empty():
                try:
                    self.voice_queue.get_nowait()
                except queue.Empty:
                    break
        
        voice_command = {
            'type': 'custom',
            'text': text
        }
        
        self.voice_queue.put(voice_command)
    
    def _speak_pose_announcement(self, command: Dict):
        """Generate and speak pose announcement."""
        pose_name = command['pose_name']
        sanskrit_name = command.get('sanskrit_name')
        pronunciation = command.get('pronunciation')
        
        # Get announcement template
        template = self.templates['pose_announcement']
        
        if pronunciation:
            text = template['with_pronunciation'].format(
                pronunciation=pronunciation,
                pose_name=pose_name.replace('_', ' ').title()
            )
        elif sanskrit_name:
            text = template['with_sanskrit'].format(
                sanskrit_name=sanskrit_name,
                pose_name=pose_name.replace('_', ' ').title()
            )
        else:
            text = template['basic'].format(
                pose_name=pose_name.replace('_', ' ').title()
            )
        
        self._speak_text(text)
    
    def _speak_correction(self, command: Dict):
        """Generate and speak correction guidance."""
        corrections = command['corrections']
        pose_name = command.get('pose_name', '')
        
        # Select most important corrections (limit to 2 for clarity)
        primary_corrections = corrections[:2]
        
        if len(primary_corrections) == 1:
            text = f"Adjust your pose: {primary_corrections[0]}"
        else:
            text = "Adjust your pose: " + ", and ".join(primary_corrections)
        
        self._speak_text(text)
    
    def _speak_transition(self, command: Dict):
        """Generate and speak transition guidance."""
        from_pose = command['from_pose'].replace('_', ' ').title()
        to_pose = command['to_pose'].replace('_', ' ').title()
        hold_time = command.get('hold_time')
        
        if hold_time:
            text = f"Hold {from_pose} for {hold_time} more seconds, then move to {to_pose}"
        else:
            text = f"Good work in {from_pose}. Now transition to {to_pose}"
        
        self._speak_text(text)
    
    def _speak_encouragement(self, command: Dict):
        """Generate and speak encouragement."""
        context = command['context']
        
        encouragement_messages = {
            'general': [
                "Great job! Keep focusing on your breath.",
                "Excellent form! Stay present in your practice.",
                "Beautiful pose! Feel the strength in your body.",
                "Wonderful! Remember to breathe deeply."
            ],
            'improvement': [
                "Much better! You're improving nicely.",
                "Great adjustment! That's the right form.",
                "Perfect! You've found the proper alignment.",
                "Excellent correction! Keep it up."
            ],
            'achievement': [
                "Outstanding! You've mastered this pose.",
                "Incredible progress! Your practice is strong.",
                "Amazing work! You're becoming more flexible.",
                "Brilliant! Your balance has improved so much."
            ]
        }
        
        import random
        messages = encouragement_messages.get(context, encouragement_messages['general'])
        text = random.choice(messages)
        
        self._speak_text(text)
    
    def _speak_text(self, text: str):
        """Speak the given text using TTS engine."""
        if not self.enabled or self.tts_engine is None:
            return
        
        try:
            logger.debug(f"Speaking: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"Error speaking text: {e}")
    
    def _load_voice_templates(self) -> Dict:
        """Load voice feedback templates."""
        default_templates = {
            "pose_announcement": {
                "basic": "You're in {pose_name}. Great work!",
                "with_sanskrit": "You're in {sanskrit_name}. {pose_name}. Well done!",
                "with_pronunciation": "You're in {pronunciation}. Beautiful pose!"
            },
            "corrections": {
                "alignment": "Focus on aligning your {body_part}.",
                "breathing": "Remember to breathe deeply and evenly.",
                "balance": "Engage your core for better balance.",
                "flexibility": "Move slowly and don't force the stretch."
            },
            "transitions": {
                "hold": "Hold this pose for {seconds} more seconds.",
                "move": "Now transition slowly to {next_pose}.",
                "rest": "Take a moment to rest and breathe."
            }
        }
        
        # Try to load custom templates if they exist
        templates_path = Path("yoga_ai_trainer/backend/data/voice_templates.json")
        if templates_path.exists():
            try:
                with templates_path.open() as f:
                    custom_templates = json.load(f)
                # Merge with defaults
                default_templates.update(custom_templates)
            except Exception as e:
                logger.warning(f"Could not load custom templates: {e}")
        
        return default_templates
    
    def set_enabled(self, enabled: bool):
        """Enable or disable voice feedback."""
        self.enabled = enabled
        logger.info(f"Voice feedback {'enabled' if enabled else 'disabled'}")
    
    def is_enabled(self) -> bool:
        """Check if voice feedback is enabled."""
        return self.enabled
    
    def clear_queue(self):
        """Clear all pending voice commands."""
        while not self.voice_queue.empty():
            try:
                self.voice_queue.get_nowait()
            except queue.Empty:
                break
    
    def shutdown(self):
        """Shutdown the voice feedback system."""
        self.voice_queue.put(None)  # Shutdown signal
        if self.voice_thread.is_alive():
            self.voice_thread.join(timeout=2.0)
        
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        
        logger.info("Voice feedback system shutdown")

# Global voice feedback instance
_voice_feedback_instance = None

def get_voice_feedback() -> VoiceFeedbackSystem:
    """Get or create the global voice feedback instance."""
    global _voice_feedback_instance
    if _voice_feedback_instance is None:
        _voice_feedback_instance = VoiceFeedbackSystem()
    return _voice_feedback_instance

def cleanup_voice_feedback():
    """Cleanup the global voice feedback instance."""
    global _voice_feedback_instance
    if _voice_feedback_instance:
        _voice_feedback_instance.shutdown()
        _voice_feedback_instance = None
