"""
🇩🇪 GERMAN FEMALE VOICE für Kira
Bark History Prompts für deutsche weibliche Stimmen
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# 🎭 GERMAN FEMALE VOICE PRESETS
GERMAN_FEMALE_VOICES = {
    "young_friendly": {
        "name": "Julia - Junge freundliche Stimme",
        "age_range": "20-30",
        "style": "friendly, energetic, clear",
        "use_cases": ["assistant", "tutorial", "friendly_chat"],
        "speaker_id": "v2/de/speaker_0"
    },
    "professional": {
        "name": "Dr. Schmidt - Professionelle Stimme", 
        "age_range": "30-40",
        "style": "professional, calm, authoritative",
        "use_cases": ["business", "presentations", "formal"],
        "speaker_id": "v2/de/speaker_1"
    },
    "warm_mature": {
        "name": "Anna - Warme reife Stimme",
        "age_range": "35-45", 
        "style": "warm, motherly, comforting",
        "use_cases": ["storytelling", "comfort", "guidance"],
        "speaker_id": "v2/de/speaker_2"
    },
    "tech_savvy": {
        "name": "Lisa - Tech-versierte Stimme",
        "age_range": "25-35",
        "style": "modern, tech-savvy, efficient", 
        "use_cases": ["ai_assistant", "tech_support", "gaming"],
        "speaker_id": "v2/de/speaker_3"
    },
    "elegant": {
        "name": "Sophie - Elegante Stimme",
        "age_range": "30-40",
        "style": "elegant, sophisticated, cultured",
        "use_cases": ["luxury", "culture", "formal_events"],
        "speaker_id": "v2/de/speaker_4"
    },
    "energetic": {
        "name": "Mia - Energetische Stimme", 
        "age_range": "22-28",
        "style": "energetic, enthusiastic, dynamic",
        "use_cases": ["sports", "motivation", "entertainment"],
        "speaker_id": "v2/de/speaker_5"
    },
    "kira_default": {
        "name": "Kira - KI-Assistentin",
        "age_range": "25-30", 
        "style": "intelligent, helpful, personable",
        "use_cases": ["ai_assistant", "personal_assistant", "companion"],
        "speaker_id": "v2/de/speaker_6"  # ✨ KIRA'S SIGNATURE VOICE
    }
}

# 🎪 EMOTION MODULATION für deutsche Stimmen
GERMAN_EMOTION_MODIFIERS = {
    "neutral": {
        "pitch_shift": 0.0,
        "speed_factor": 1.0,
        "energy_level": 0.5,
        "warmth": 0.5
    },
    "happy": {
        "pitch_shift": 0.1,
        "speed_factor": 1.1,
        "energy_level": 0.8,
        "warmth": 0.7
    },
    "excited": {
        "pitch_shift": 0.15,
        "speed_factor": 1.2,
        "energy_level": 0.9,
        "warmth": 0.8
    },
    "calm": {
        "pitch_shift": -0.05,
        "speed_factor": 0.9,
        "energy_level": 0.3,
        "warmth": 0.6
    },
    "serious": {
        "pitch_shift": -0.1,
        "speed_factor": 0.95,
        "energy_level": 0.4,
        "warmth": 0.3
    },
    "concerned": {
        "pitch_shift": -0.05,
        "speed_factor": 0.9,
        "energy_level": 0.6,
        "warmth": 0.7
    },
    "friendly": {
        "pitch_shift": 0.05,
        "speed_factor": 1.05,
        "energy_level": 0.7,
        "warmth": 0.8
    },
    "professional": {
        "pitch_shift": -0.02,
        "speed_factor": 1.0,
        "energy_level": 0.5,
        "warmth": 0.4
    }
}

# 🎯 KIRA PERSONALITY VOICE MAPPING
KIRA_VOICE_CONTEXTS = {
    "greeting": {
        "voice": "kira_default",
        "emotion": "friendly",
        "sample_texts": [
            "Hallo! Ich bin Kira, deine KI-Assistentin.",
            "Schön, dass du da bist!",
            "Was kann ich heute für dich tun?"
        ]
    },
    "helping": {
        "voice": "kira_default", 
        "emotion": "helpful",
        "sample_texts": [
            "Lass mich dir dabei helfen.",
            "Das kann ich für dich erledigen.",
            "Kein Problem, das kriegen wir hin!"
        ]
    },
    "thinking": {
        "voice": "kira_default",
        "emotion": "neutral",
        "sample_texts": [
            "Moment, ich denke darüber nach...",
            "Das ist eine interessante Frage.",
            "Lass mich das analysieren."
        ]
    },
    "error": {
        "voice": "kira_default",
        "emotion": "concerned", 
        "sample_texts": [
            "Entschuldigung, da ist etwas schiefgelaufen.",
            "Das hat leider nicht funktioniert.",
            "Lass mich das nochmal versuchen."
        ]
    },
    "success": {
        "voice": "kira_default",
        "emotion": "happy",
        "sample_texts": [
            "Perfekt! Das hat geklappt.",
            "Erledigt!",
            "Super, alles fertig!"
        ]
    }
}

class GermanFemaleVoiceManager:
    """Manager für deutsche weibliche Stimmen"""
    
    def __init__(self):
        self.current_voice = "kira_default"
        self.current_emotion = "neutral"
        self.voice_cache = {}
        
        logger.info("🇩🇪 German Female Voice Manager initialized")
    
    def get_voice_preset(self, voice_name: str = None) -> Dict:
        """Hole Voice Preset"""
        if voice_name is None:
            voice_name = self.current_voice
        
        if voice_name not in GERMAN_FEMALE_VOICES:
            logger.warning(f"⚠️ Unknown voice '{voice_name}', using default")
            voice_name = "kira_default"
        
        return GERMAN_FEMALE_VOICES[voice_name]
    
    def get_emotion_modifier(self, emotion: str = None) -> Dict:
        """Hole Emotion Modifier"""
        if emotion is None:
            emotion = self.current_emotion
        
        if emotion not in GERMAN_EMOTION_MODIFIERS:
            logger.warning(f"⚠️ Unknown emotion '{emotion}', using neutral")
            emotion = "neutral"
        
        return GERMAN_EMOTION_MODIFIERS[emotion]
    
    def get_voice_for_context(self, context: str) -> Tuple[str, str]:
        """Hole Voice und Emotion für Context"""
        if context in KIRA_VOICE_CONTEXTS:
            ctx = KIRA_VOICE_CONTEXTS[context]
            return ctx["voice"], ctx["emotion"]
        else:
            return self.current_voice, self.current_emotion
    
    def set_voice(self, voice_name: str):
        """Setze aktuelle Stimme"""
        if voice_name in GERMAN_FEMALE_VOICES:
            self.current_voice = voice_name
            logger.info(f"🎤 Voice set to: {GERMAN_FEMALE_VOICES[voice_name]['name']}")
        else:
            logger.warning(f"⚠️ Unknown voice '{voice_name}'")
    
    def set_emotion(self, emotion: str):
        """Setze aktuelle Emotion"""
        if emotion in GERMAN_EMOTION_MODIFIERS:
            self.current_emotion = emotion
            logger.info(f"😊 Emotion set to: {emotion}")
        else:
            logger.warning(f"⚠️ Unknown emotion '{emotion}'")
    
    def get_available_voices(self) -> List[Dict]:
        """Liste verfügbare Stimmen"""
        return [
            {
                "id": voice_id,
                "name": voice_data["name"],
                "style": voice_data["style"],
                "use_cases": voice_data["use_cases"],
                "speaker_id": voice_data["speaker_id"]
            }
            for voice_id, voice_data in GERMAN_FEMALE_VOICES.items()
        ]
    
    def get_available_emotions(self) -> List[str]:
        """Liste verfügbare Emotionen"""
        return list(GERMAN_EMOTION_MODIFIERS.keys())
    
    def generate_voice_config(self, voice_name: str = None, emotion: str = None) -> Dict:
        """Generiere Voice Config für Bark"""
        voice_preset = self.get_voice_preset(voice_name)
        emotion_modifier = self.get_emotion_modifier(emotion)
        
        return {
            "speaker_id": voice_preset["speaker_id"],
            "voice_preset": voice_preset,
            "emotion_modifier": emotion_modifier,
            "language": "de",
            "gender": "female",
            "style": voice_preset["style"]
        }

# 🎭 HISTORY PROMPTS für Bark (falls benötigt)
def create_german_female_history_prompts():
    """Erstelle History Prompts für deutsche weibliche Stimmen"""
    
    # Basis Audio Features für deutsche weibliche Stimmen
    base_features = {
        "sample_rate": 24000,
        "language": "de",
        "gender": "female",
        "accent": "standard_german"
    }
    
    # Mock History Prompts (in der Praxis würden echte Audio-Features verwendet)
    history_prompts = {}
    
    for voice_id, voice_data in GERMAN_FEMALE_VOICES.items():
        history_prompts[voice_data["speaker_id"]] = {
            **base_features,
            "voice_characteristics": {
                "age_range": voice_data["age_range"],
                "style": voice_data["style"],
                "name": voice_data["name"]
            },
            # Placeholder für echte Audio-Features
            "semantic_prompt": f"german_female_{voice_id}_semantic",
            "coarse_prompt": f"german_female_{voice_id}_coarse", 
            "fine_prompt": f"german_female_{voice_id}_fine"
        }
    
    return history_prompts

# 🌟 KIRA SPECIFIC VOICE FUNCTIONS
def get_kira_voice_config(emotion: str = "neutral") -> Dict:
    """Hole Kira's spezifische Voice Config"""
    manager = GermanFemaleVoiceManager()
    return manager.generate_voice_config("kira_default", emotion)

def get_context_voice_config(context: str) -> Dict:
    """Hole Voice Config für spezifischen Context"""
    manager = GermanFemaleVoiceManager()
    voice, emotion = manager.get_voice_for_context(context)
    return manager.generate_voice_config(voice, emotion)

# 🧪 TEST FUNCTIONS
def test_german_voices():
    """Teste deutsche Stimmen"""
    print("🇩🇪 TESTING GERMAN FEMALE VOICES")
    print("=" * 50)
    
    manager = GermanFemaleVoiceManager()
    
    print("\n📋 Available Voices:")
    for voice in manager.get_available_voices():
        print(f"   🎤 {voice['name']}")
        print(f"      Style: {voice['style']}")
        print(f"      Use cases: {', '.join(voice['use_cases'])}")
        print(f"      Speaker ID: {voice['speaker_id']}")
        print()
    
    print("🎭 Available Emotions:")
    for emotion in manager.get_available_emotions():
        print(f"   😊 {emotion}")
    
    print("\n🎯 Kira Contexts:")
    for context, config in KIRA_VOICE_CONTEXTS.items():
        print(f"   {context}: {config['voice']} + {config['emotion']}")
    
    print("\n✅ German Female Voice System ready!")

if __name__ == "__main__":
    test_german_voices()