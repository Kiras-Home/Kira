# voice/synthesis/phonetic_processors.py
"""
Phonetic Processors für Voice Synthesis
"""
import logging
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PhoneticRule:
    """Phonetic transformation rule"""
    pattern: str
    replacement: str
    context: str = ""

class GermanPhoneticProcessor:
    """
    German Phonetic Processor for better TTS pronunciation
    """
    
    def __init__(self):
        self.phonetic_rules = self._initialize_german_rules()
        
    def _initialize_german_rules(self) -> List[PhoneticRule]:
        """Initialize German phonetic rules"""
        return [
            # Umlaute
            PhoneticRule(r'ä', 'ae'),
            PhoneticRule(r'ö', 'oe'), 
            PhoneticRule(r'ü', 'ue'),
            PhoneticRule(r'ß', 'ss'),
            
            # Ch-Laute
            PhoneticRule(r'ch(?=[ei])', 'sch'),
            PhoneticRule(r'ch(?=[aou])', 'k'),
            
            # Finale Konsonanten
            PhoneticRule(r'ig$', 'ich'),
            PhoneticRule(r'ng', 'nk'),
            
            # S-Laute
            PhoneticRule(r'st(?=[aeiou])', 'scht'),
            PhoneticRule(r'sp(?=[aeiou])', 'schp'),
        ]
    
    def process_text(self, text: str) -> str:
        """Process text with German phonetic rules"""
        try:
            processed = text.lower()
            
            for rule in self.phonetic_rules:
                processed = re.sub(rule.pattern, rule.replacement, processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Phonetic processing error: {e}")
            return text