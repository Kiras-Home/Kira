"""
Cross-Platform Utilities - Extrahierte Funktionen aus Personal Identity System
Für Enhanced Memory Database und Human-Like Memory System optimiert
"""

import re
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class CrossPlatformRecognition:
    """Cross-Platform User Recognition für Enhanced Memory System"""
    
    def __init__(self, memory_database):
        """
        Args:
            memory_database: EnhancedMemoryDatabase instance
        """
        if not hasattr(memory_database, 'db_path'):
            raise TypeError(f"Expected EnhancedMemoryDatabase instance, got {type(memory_database)}")
        
        self.memory_database = memory_database
        self.data_dir = memory_database.data_dir
        
        """
        Args:
            memory_database: EnhancedMemoryDatabase instance
        """
        if not hasattr(memory_database, 'db_path'):
            raise TypeError(f"Expected EnhancedMemoryDatabase instance, got {type(memory_database)}")
        
        self.memory_database = memory_database
        self.data_dir = memory_database.data_dir
        
        self.name_patterns = [
            # Direkte Einführung
            r"(?:ich\s+(?:heiße|bin)|mein\s+name\s+ist|ich\s+bin)\s+([a-zA-ZäöüÄÖÜß]+)",
            r"(?:ich\s+bin\s+der|ich\s+bin\s+die)?\s*([a-zA-ZäöüÄÖÜß]+)",
            
            # Englische Varianten
            r"(?:my\s+name\s+is|i\s+am|i'm)\s+([a-zA-Z]+)",
            r"(?:call\s+me|name\s+is)\s+([a-zA-Z]+)",
            
            # Begrüßungen mit Namen
            r"hallo\s*,?\s*ich\s+bin\s+([a-zA-ZäöüÄÖÜß]+)",
            r"hi\s*,?\s*i\s+am\s+([a-zA-Z]+)",
            
            # Informelle Varianten
            r"bin\s+(?:der|die)?\s*([a-zA-ZäöüÄÖÜß]+)",
            r"(?:das\s+bin\s+ich|das\s+ist)\s*,?\s*([a-zA-ZäöüÄÖÜß]+)"
        ]
        
        self.cross_platform_patterns = [
            # Explizite Cross-Platform Referenzen
            r"(?:du\s+kennst\s+mich\s+(?:schon\s+)?(?:von|aus)|wir\s+haben\s+(?:schon\s+)?(?:mal\s+)?(?:geredet|gesprochen))",
            r"(?:ich\s+war\s+(?:schon\s+)?(?:mal\s+)?hier|wir\s+hatten\s+(?:schon\s+)?contact)",
            r"(?:erinnerst\s+du\s+dich\s+an\s+mich|kennst\s+du\s+mich\s+(?:noch|schon))",
            r"(?:ich\s+bin\s+(?:der|die)\s+(?:gleiche|selbe)|same\s+person)",
            
            # Platform-spezifische Referenzen
            r"(?:vom\s+(?:handy|smartphone|telefon|tablet)|from\s+(?:phone|mobile|tablet))",
            r"(?:über\s+(?:alexa|google|assistant)|through\s+(?:alexa|google|assistant))",
            r"(?:am\s+(?:computer|laptop|pc)|on\s+(?:computer|laptop|pc))",
            
            # Memory-Referenzen
            r"(?:du\s+(?:weißt|kennst)\s+(?:doch|ja)|you\s+(?:know|remember))",
            r"(?:wie\s+letztes\s+mal|like\s+last\s+time)",
            r"(?:wir\s+haben\s+über\s+.*\s+gesprochen|we\s+talked\s+about)"
        ]
    
    def extract_name_from_message(self, message: str) -> Optional[str]:
        """
        Extrahiert Namen aus Nachrichten mit verbesserter Erkennung
        
        Args:
            message: User-Nachricht
            
        Returns:
            Extrahierter Name oder None
        """
        
        if not message:
            return None
        
        # Normalisiere Input
        message_clean = message.strip().lower()
        
        # Prüfe alle Patterns
        for pattern in self.name_patterns:
            match = re.search(pattern, message_clean, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                
                # Validiere Namen
                if self._is_valid_name(name):
                    # Capitalize properly
                    return name.capitalize()
        
        return None
    
    def detect_cross_platform_introduction(self, message: str) -> Dict[str, Any]:
        """
        Erkennt Cross-Platform Referenzen in Nachrichten
        
        Args:
            message: User-Nachricht
            
        Returns:
            Dictionary mit Erkennungsresultaten
        """
        
        result = {
            'has_cross_platform_reference': False,
            'confidence': 0.0,
            'detected_patterns': [],
            'suggested_platform': None,
            'memory_reference': False
        }
        
        if not message:
            return result
        
        message_clean = message.strip().lower()
        confidence_score = 0.0
        detected_patterns = []
        
        # Prüfe Cross-Platform Patterns
        for pattern in self.cross_platform_patterns:
            if re.search(pattern, message_clean, re.IGNORECASE):
                detected_patterns.append(pattern)
                confidence_score += 0.2
        
        # Platform-spezifische Erkennung
        platform_hints = {
            'mobile': ['handy', 'smartphone', 'telefon', 'phone', 'mobile'],
            'voice_assistant': ['alexa', 'google', 'assistant', 'sprachassistent'],
            'desktop': ['computer', 'laptop', 'pc', 'desktop'],
            'tablet': ['tablet', 'ipad']
        }
        
        suggested_platform = None
        for platform, keywords in platform_hints.items():
            for keyword in keywords:
                if keyword in message_clean:
                    suggested_platform = platform
                    confidence_score += 0.1
                    break
        
        # Memory-Referenz Erkennung
        memory_keywords = [
            'erinnerst', 'weißt', 'kennst', 'gesprochen', 'geredet', 
            'letztes mal', 'remember', 'know', 'talked', 'last time'
        ]
        
        memory_reference = any(keyword in message_clean for keyword in memory_keywords)
        if memory_reference:
            confidence_score += 0.15
        
        # Finalisiere Ergebnis
        result.update({
            'has_cross_platform_reference': confidence_score > 0.1,
            'confidence': min(1.0, confidence_score),
            'detected_patterns': detected_patterns,
            'suggested_platform': suggested_platform,
            'memory_reference': memory_reference
        })
        
        return result
    
    def _is_valid_name(self, name: str) -> bool:
        """Validiert extrahierte Namen"""
        
        if not name or len(name) < 2:
            return False
        
        # Ausschluss von häufigen False Positives
        invalid_names = {
            'ich', 'du', 'er', 'sie', 'wir', 'ihr', 'kira',
            'assistant', 'ai', 'bot', 'system', 'user',
            'ja', 'nein', 'ok', 'okay', 'danke', 'bitte',
            'i', 'you', 'he', 'she', 'we', 'they',
            'yes', 'no', 'ok', 'okay', 'thanks', 'please'
        }
        
        if name.lower() in invalid_names:
            return False
        
        # Nur Buchstaben (mit deutschen Umlauten)
        if not re.match(r'^[a-zA-ZäöüÄÖÜß]+$', name):
            return False
        
        return True
    
    def analyze_introduction_context(self, message: str, session_context: Dict = None) -> Dict[str, Any]:
        """
        Analysiert Einführungskontext für bessere User Recognition
        
        Args:
            message: User-Nachricht
            session_context: Optional session context
            
        Returns:
            Umfassende Kontext-Analyse
        """
        
        analysis = {
            'extracted_name': None,
            'cross_platform_signals': {},
            'introduction_type': 'unknown',
            'confidence_level': 'low',
            'recommended_action': 'store_as_new',
            'context_clues': []
        }
        
        # Name extrahieren
        extracted_name = self.extract_name_from_message(message)
        if extracted_name:
            analysis['extracted_name'] = extracted_name
            analysis['context_clues'].append(f'name_detected: {extracted_name}')
        
        # Cross-Platform Signale
        cross_platform = self.detect_cross_platform_introduction(message)
        analysis['cross_platform_signals'] = cross_platform
        
        # Introduction Type bestimmen
        message_lower = message.lower()
        
        if extracted_name and not cross_platform['has_cross_platform_reference']:
            analysis['introduction_type'] = 'new_introduction'
            analysis['recommended_action'] = 'create_new_user'
            
        elif cross_platform['has_cross_platform_reference']:
            analysis['introduction_type'] = 'cross_platform_recognition'
            analysis['recommended_action'] = 'search_existing_user'
            
        elif any(word in message_lower for word in ['hallo', 'hi', 'hey']):
            analysis['introduction_type'] = 'greeting_only'
            analysis['recommended_action'] = 'request_introduction'
        
        # Confidence Level
        total_confidence = 0.0
        
        if extracted_name:
            total_confidence += 0.4
        
        if cross_platform['confidence'] > 0.3:
            total_confidence += cross_platform['confidence'] * 0.6
        
        if total_confidence > 0.7:
            analysis['confidence_level'] = 'high'
        elif total_confidence > 0.4:
            analysis['confidence_level'] = 'medium'
        else:
            analysis['confidence_level'] = 'low'
        
        # Session Context berücksichtigen
        if session_context:
            if session_context.get('previous_interactions', 0) > 0:
                analysis['context_clues'].append('returning_session')
                analysis['recommended_action'] = 'check_session_history'
        
        return analysis

class EnhancedUserMatcher:
    """Enhanced User Matching für Memory Database Integration"""
    
    def __init__(self, memory_database):
        """
        Args:
            memory_database: EnhancedMemoryDatabase instance
        """
        if not hasattr(memory_database, 'db_path'):
            raise TypeError(f"Expected EnhancedMemoryDatabase instance, got {type(memory_database)}")
        
        self.memory_database = memory_database
        self.data_dir = memory_database.data_dir
        
        self.memory_database = memory_database
        self.recognition = CrossPlatformRecognition(memory_database)
    
    def find_user_by_cross_platform_signals(
        self, 
        message: str, 
        device_context: str = None,
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Findet User basierend auf Cross-Platform Signalen in Enhanced Memory Database
        
        Args:
            message: User-Nachricht
            device_context: Device-Kontext
            session_id: Session ID
            
        Returns:
            Matching-Ergebnisse mit User-Kandidaten
        """
        
        # Analysiere Nachricht
        analysis = self.recognition.analyze_introduction_context(message)
        
        matching_result = {
            'analysis': analysis,
            'user_candidates': [],
            'matching_confidence': 0.0,
            'recommended_user_id': None,
            'create_new_user': False
        }
        
        try:
            from ..storage.memory_database import MemorySearchFilter
            
            # Suche nach ähnlichen Memories wenn Name extrahiert
            if analysis['extracted_name']:
                
                # Suche in Memory Database nach Namen
                search_filter = MemorySearchFilter(
                    query=analysis['extracted_name'],
                    memory_type='conversation',
                    limit=20
                )
                
                similar_memories = self.memory_database.search_memories(search_filter)
                
                # Analysiere gefundene Memories
                user_scores = {}
                for memory in similar_memories:
                    user_id = memory['user_id']
                    
                    # Score basierend auf Name-Match
                    content_lower = memory['content'].lower()
                    name_lower = analysis['extracted_name'].lower()
                    
                    score = 0.0
                    if name_lower in content_lower:
                        score += 0.5
                    
                    # Device Context Match
                    if device_context and memory.get('device_context') == device_context:
                        score += 0.3
                    
                    # Recency Score
                    memory_age_days = (datetime.now() - datetime.fromisoformat(memory['created_at'])).days
                    recency_score = max(0.0, 1.0 - memory_age_days / 30.0)  # 30 Tage Fenster
                    score += recency_score * 0.2
                    
                    if user_id not in user_scores:
                        user_scores[user_id] = {'score': 0.0, 'memories': 0, 'name_matches': 0}
                    
                    user_scores[user_id]['score'] += score
                    user_scores[user_id]['memories'] += 1
                    
                    if name_lower in content_lower:
                        user_scores[user_id]['name_matches'] += 1
                
                # Sortiere Kandidaten nach Score
                candidates = []
                for user_id, data in user_scores.items():
                    avg_score = data['score'] / data['memories'] if data['memories'] > 0 else 0.0
                    
                    candidates.append({
                        'user_id': user_id,
                        'confidence': avg_score,
                        'memory_count': data['memories'],
                        'name_matches': data['name_matches'],
                        'match_reason': 'name_and_context'
                    })
                
                candidates.sort(key=lambda x: x['confidence'], reverse=True)
                matching_result['user_candidates'] = candidates[:5]  # Top 5
                
                # Beste Match bestimmen
                if candidates and candidates[0]['confidence'] > 0.6:
                    matching_result['recommended_user_id'] = candidates[0]['user_id']
                    matching_result['matching_confidence'] = candidates[0]['confidence']
                else:
                    matching_result['create_new_user'] = True
            
            # Cross-Platform Suche wenn starke Signale
            elif analysis['cross_platform_signals']['confidence'] > 0.5:
                
                # Suche nach ähnlichen Cross-Platform Patterns
                search_filter = MemorySearchFilter(
                    device_context=device_context,
                    memory_type='conversation',
                    limit=10
                )
                
                device_memories = self.memory_database.search_memories(search_filter)
                
                # Analysiere Device-History für mögliche User
                if device_memories:
                    user_frequency = {}
                    for memory in device_memories:
                        user_id = memory['user_id']
                        user_frequency[user_id] = user_frequency.get(user_id, 0) + 1
                    
                    # Häufigste User für dieses Device
                    most_frequent = max(user_frequency.items(), key=lambda x: x[1])
                    
                    if most_frequent[1] > 2:  # Mindestens 3 Interactions
                        matching_result['user_candidates'].append({
                            'user_id': most_frequent[0],
                            'confidence': 0.4,  # Moderate Confidence
                            'memory_count': most_frequent[1],
                            'match_reason': 'device_history'
                        })
                        
                        matching_result['recommended_user_id'] = most_frequent[0]
                        matching_result['matching_confidence'] = 0.4
            
            # Default: Neue User erstellen
            if not matching_result['recommended_user_id']:
                matching_result['create_new_user'] = True
                
                # Generiere User ID basierend auf verfügbaren Infos
                if analysis['extracted_name']:
                    suggested_user_id = f"{analysis['extracted_name'].lower()}_{device_context or 'unknown'}"
                else:
                    suggested_user_id = f"user_{session_id[:8] if session_id else 'anonymous'}"
                
                matching_result['suggested_user_id'] = suggested_user_id
                
        except Exception as e:
            logger.error(f"❌ User Matching Fehler: {e}")
            matching_result['create_new_user'] = True
            matching_result['error'] = str(e)
        
        return matching_result
    
    def create_user_introduction_memory(
        self,
        user_id: str,
        session_id: str,
        introduction_message: str,
        device_context: str = None,
        analysis: Dict = None
    ) -> Optional[int]:
        """
        Erstellt Introduction Memory in Enhanced Database
        
        Args:
            user_id: User ID
            session_id: Session ID  
            introduction_message: Original introduction message
            device_context: Device context
            analysis: Recognition analysis results
            
        Returns:
            Memory ID oder None
        """
        
        try:
            # Enhanced Memory Entry für Introduction
            memory_id = self.memory_database.store_enhanced_memory(
                session_id=session_id,
                user_id=user_id,
                memory_type='conversation',
                content=introduction_message,
                metadata={
                    'introduction': True,
                    'extracted_name': analysis.get('extracted_name') if analysis else None,
                    'cross_platform_signals': analysis.get('cross_platform_signals') if analysis else {},
                    'recognition_analysis': analysis
                },
                importance=8,  # High importance for introductions
                tags=['introduction', 'user_identity', 'first_contact'],
                device_context=device_context,
                intent_detected='user_introduction',
                user_context='introduction_phase',
                memory_category='identity',
                learning_weight=1.5,  # High learning weight
                attention_weight=0.9,
                stm_activation_level=0.8,
                ltm_significance_score=0.7
            )
            
            logger.info(f"✅ User Introduction Memory erstellt: {memory_id} für User: {user_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"❌ Introduction Memory Creation Fehler: {e}")
            return None

# === CONVENIENCE FUNCTIONS ===

def extract_name(message: str) -> Optional[str]:
    """Convenience function für Name Extraction"""
    recognition = CrossPlatformRecognition()
    return recognition.extract_name_from_message(message)

def detect_cross_platform_reference(message: str) -> bool:
    """Convenience function für Cross-Platform Detection"""
    recognition = CrossPlatformRecognition()
    result = recognition.detect_cross_platform_introduction(message)
    return result['has_cross_platform_reference']

def analyze_user_introduction(message: str, session_context: Dict = None) -> Dict[str, Any]:
    """Convenience function für vollständige Introduction Analysis"""
    recognition = CrossPlatformRecognition()
    return recognition.analyze_introduction_context(message, session_context)

# Export functions
__all__ = [
    'CrossPlatformRecognition',
    'EnhancedUserMatcher', 
    'extract_name',
    'detect_cross_platform_reference',
    'analyze_user_introduction'
]