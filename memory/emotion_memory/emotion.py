"""
Emotion Memory - Speichert und analysiert emotionale Daten
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import Counter

from ..storage.memory_database import MemoryDatabase

logger = logging.getLogger(__name__)

class EmotionMemory:
    """Emotionales Gedächtnis für Kira"""
    
    def __init__(self, database: MemoryDatabase):
        self.db = database
        
    def store_emotion(
        self,
        user_id: str,
        emotion_type: str,
        intensity: float = 0.5,
        context: str = "",
        memory_entry_id: Optional[int] = None
    ) -> Optional[int]:
        """Speichert eine emotionale Erfahrung"""
        
        try:
            # Verwende bestehende memory_entries Tabelle mit emotion memory_type
            
            emotion_entry = MemoryEntry(
                session_id="emotions",
                user_id=user_id,
                memory_type="emotion",
                content=f"Emotion: {emotion_type} (Intensität: {intensity})",
                metadata={
                    'emotion_type': emotion_type,
                    'intensity': intensity,
                    'context': context,
                    'memory_entry_id': memory_entry_id,
                    'detection_time': datetime.now().isoformat()
                },
                importance=max(5, int(intensity * 10)),  # Intensität → Wichtigkeit
                tags=[emotion_type, 'emotion_data']
            )
            
            with self.db.get_connection() as conn:
                cursor = conn.execute('''
                    INSERT INTO memory_entries 
                    (session_id, user_id, memory_type, content, metadata, importance, 
                     created_at, last_accessed, access_count, tags, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    emotion_entry.session_id,
                    emotion_entry.user_id,
                    emotion_entry.memory_type,
                    emotion_entry.content,
                    str(emotion_entry.metadata),
                    emotion_entry.importance,
                    emotion_entry.created_at.isoformat(),
                    emotion_entry.last_accessed.isoformat(),
                    emotion_entry.access_count,
                    ','.join(emotion_entry.tags),
                    self.db._generate_content_hash(emotion_entry.content)
                ))
                
                emotion_id = cursor.lastrowid
                conn.commit()
                
            logger.debug(f"😊 Emotion gespeichert: {emotion_type} ({intensity}) für {user_id}")
            return emotion_id
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Speichern der Emotion: {e}")
            return None
    
    def get_emotion_patterns(self, user_id: str = "default", days: int = 30) -> Dict[str, Any]:
        """Analysiert emotionale Muster des Benutzers"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT * FROM memory_entries 
                    WHERE user_id = ? AND memory_type = 'emotion'
                    AND created_at > ?
                    ORDER BY created_at DESC
                ''', (user_id, cutoff_date.isoformat()))
                
                emotions = cursor.fetchall()
                
                if not emotions:
                    return {'status': 'no_data', 'days_analyzed': days}
                
                # Analysiere Patterns
                emotion_types = []
                intensities = []
                daily_emotions = {}
                
                for emotion in emotions:
                    metadata = eval(emotion['metadata'])
                    emotion_type = metadata.get('emotion_type', 'unknown')
                    intensity = metadata.get('intensity', 0.5)
                    
                    emotion_types.append(emotion_type)
                    intensities.append(intensity)
                    
                    # Gruppiere nach Tagen
                    day = emotion['created_at'][:10]  # YYYY-MM-DD
                    if day not in daily_emotions:
                        daily_emotions[day] = []
                    daily_emotions[day].append(emotion_type)
                
                # Berechne Statistiken
                emotion_counts = Counter(emotion_types)
                avg_intensity = sum(intensities) / len(intensities)
                
                patterns = {
                    'user_id': user_id,
                    'analysis_period_days': days,
                    'total_emotions': len(emotions),
                    'dominant_emotions': emotion_counts.most_common(5),
                    'average_intensity': round(avg_intensity, 2),
                    'emotional_stability': self._calculate_stability(intensities),
                    'daily_patterns': self._analyze_daily_patterns(daily_emotions),
                    'emotion_diversity': len(set(emotion_types)),
                    'last_updated': datetime.now().isoformat()
                }
                
                return patterns
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Emotion-Pattern-Analyse: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_recent_emotions(
        self, 
        user_id: str = "default", 
        limit: int = 10
    ) -> List[Dict]:
        """Holt die letzten Emotionen"""
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT * FROM memory_entries 
                    WHERE user_id = ? AND memory_type = 'emotion'
                    ORDER BY created_at DESC LIMIT ?
                ''', (user_id, limit))
                
                emotions = []
                for row in cursor.fetchall():
                    metadata = eval(row['metadata'])
                    emotions.append({
                        'id': row['id'],
                        'emotion_type': metadata.get('emotion_type', 'unknown'),
                        'intensity': metadata.get('intensity', 0.5),
                        'context': metadata.get('context', ''),
                        'created_at': row['created_at'],
                        'importance': row['importance']
                    })
                
                return emotions
                
        except Exception as e:
            logger.error(f"❌ Fehler beim Holen der Emotionen: {e}")
            return []
    
    def get_emotion_for_context(
        self,
        user_id: str = "default",
        context_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Holt Emotionen für spezifischen Kontext"""
        
        if not context_keywords:
            return {}
        
        try:
            with self.db.get_connection() as conn:
                # Suche nach Emotionen mit passendem Kontext
                context_query = " OR ".join([f"metadata LIKE '%{keyword}%'" for keyword in context_keywords])
                
                cursor = conn.execute(f'''
                    SELECT * FROM memory_entries 
                    WHERE user_id = ? AND memory_type = 'emotion'
                    AND ({context_query})
                    ORDER BY created_at DESC LIMIT 5
                ''', (user_id,))
                
                emotions = cursor.fetchall()
                
                if not emotions:
                    return {'status': 'no_matching_emotions'}
                
                # Analysiere gefundene Emotionen
                emotion_types = []
                intensities = []
                
                for emotion in emotions:
                    metadata = eval(emotion['metadata'])
                    emotion_types.append(metadata.get('emotion_type', 'neutral'))
                    intensities.append(metadata.get('intensity', 0.5))
                
                return {
                    'context_keywords': context_keywords,
                    'matching_emotions': len(emotions),
                    'dominant_emotion': Counter(emotion_types).most_common(1)[0] if emotion_types else ('neutral', 0),
                    'average_intensity': round(sum(intensities) / len(intensities), 2) if intensities else 0.5,
                    'emotion_history': emotion_types
                }
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Kontext-Emotion-Suche: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def cleanup_old_entries(self, days_to_keep: int = 90) -> int:
        """Bereinigt alte Emotion-Einträge"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute('''
                    DELETE FROM memory_entries 
                    WHERE memory_type = 'emotion' 
                    AND importance < 7 
                    AND created_at < ?
                ''', (cutoff_date.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"🧹 {deleted_count} alte Emotion-Einträge gelöscht")
                return deleted_count
                
        except Exception as e:
            logger.error(f"❌ Fehler beim Bereinigen der Emotionen: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Holt Emotion-Statistiken"""
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_emotions,
                        COUNT(DISTINCT user_id) as users_with_emotions,
                        AVG(importance) as avg_importance,
                        MAX(created_at) as last_emotion
                    FROM memory_entries 
                    WHERE memory_type = 'emotion'
                ''')
                
                row = cursor.fetchone()
                return {
                    'total_emotions': row['total_emotions'],
                    'users_with_emotions': row['users_with_emotions'],
                    'avg_importance': round(row['avg_importance'] or 0, 2),
                    'last_emotion': row['last_emotion']
                }
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Emotion-Statistiken: {e}")
            return {}
    
    def _calculate_stability(self, intensities: List[float]) -> float:
        """Berechnet emotionale Stabilität (weniger Varianz = stabiler)"""
        if len(intensities) < 2:
            return 1.0
        
        avg = sum(intensities) / len(intensities)
        variance = sum((x - avg) ** 2 for x in intensities) / len(intensities)
        
        # Stabilität: 1.0 = sehr stabil, 0.0 = sehr instabil
        return max(0.0, 1.0 - variance)
    
    def _analyze_daily_patterns(self, daily_emotions: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analysiert tägliche emotionale Muster"""
        
        if not daily_emotions:
            return {}
        
        daily_dominant = {}
        emotion_frequency = Counter()
        
        for day, emotions in daily_emotions.items():
            if emotions:
                dominant = Counter(emotions).most_common(1)[0][0]
                daily_dominant[day] = dominant
                emotion_frequency.update(emotions)
        
        return {
            'days_with_data': len(daily_emotions),
            'most_frequent_daily_emotion': emotion_frequency.most_common(1)[0] if emotion_frequency else ('neutral', 0),
            'emotional_consistency': len(set(daily_dominant.values())) / len(daily_dominant) if daily_dominant else 0
        }