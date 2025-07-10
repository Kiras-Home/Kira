"""
Enhanced Search Utilities - Erweiterte Suchfunktionen für Memory Database
Integriert semantische Suche, Ähnlichkeitsanalyse und temporale Gruppierung
"""

import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import Counter

logger = logging.getLogger(__name__)

class MemorySearchEnhancer:
    """Erweiterte Suchfunktionen für Enhanced Memory Database"""
    
    def __init__(self, memory_database):
        self.memory_database = memory_database
        
        # Enhanced Search Categories
        self.search_categories = {
            'personal_info': ['name', 'alter', 'geburtstag', 'birthday', 'wohne', 'arbeite', 'beruf'],
            'preferences': ['mag', 'liebe', 'hasse', 'interessiert', 'hobby', 'gefällt', 'favorit'],
            'technical': ['code', 'programm', 'computer', 'system', 'software', 'entwicklung'],
            'smart_home': ['licht', 'light', 'sensor', 'gerät', 'device', 'automation', 'steuerung'],
            'emotional': ['freude', 'traurig', 'glücklich', 'wütend', 'frustration', 'emotion', 'gefühl'],
            'temporal': ['heute', 'gestern', 'morgen', 'letzte woche', 'nächste', 'damals'],
            'relationship': ['familie', 'freund', 'kollege', 'partner', 'beziehung', 'sozial'],
            'learning': ['lernen', 'verstehen', 'erklärung', 'wissen', 'information', 'bildung']
        }
        
        # Synonym Mappings
        self.synonyms = {
            'happy': ['glücklich', 'freude', 'froh', 'zufrieden'],
            'sad': ['traurig', 'niedergeschlagen', 'melancholisch'],
            'angry': ['wütend', 'sauer', 'verärgert', 'frustriert'],
            'home': ['zuhause', 'heim', 'wohnung', 'haus'],
            'work': ['arbeit', 'job', 'beruf', 'arbeitsplatz']
        }
    
    def enhanced_semantic_search(
        self,
        query: str,
        user_id: str = "default",
        search_filter: Optional[object] = None,
        enhance_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Enhanced Semantic Search mit Category Detection und Synonym Expansion
        """
        
        try:
            # Detect search category
            detected_category = self._detect_search_category(query)
            
            # Expand query with synonyms
            expanded_query = self._expand_query_with_synonyms(query, detected_category)
            
            # Use existing Enhanced Memory Database search
            from ..storage.memory_database import MemorySearchFilter  # ✅ KORRIGIERT
            
            if search_filter is None:
                search_filter = MemorySearchFilter(
                    query=expanded_query,
                    user_id=user_id,
                    limit=50  # Get more results for post-processing
                )
            else:
                # Update existing filter with expanded query
                search_filter.query = expanded_query
            
            # Get base results from Enhanced Memory Database
            base_results = self.memory_database.search_memories(search_filter)
            
            if not enhance_results:
                return base_results
            
            # Enhance results with semantic analysis
            enhanced_results = []
            for result in base_results:
                enhanced_result = result.copy()
                
                # Calculate enhanced semantic score
                semantic_score = self._calculate_enhanced_semantic_score(
                    query, result, detected_category
                )
                enhanced_result['enhanced_semantic_score'] = semantic_score
                
                # Add category information
                enhanced_result['detected_category'] = detected_category
                
                # Add keyword matches
                keyword_matches = self._find_keyword_matches(query, result.get('content', ''))
                enhanced_result['keyword_matches'] = keyword_matches
                
                enhanced_results.append(enhanced_result)
            
            # Sort by enhanced semantic score
            enhanced_results.sort(
                key=lambda x: x.get('enhanced_semantic_score', 0), 
                reverse=True
            )
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"❌ Enhanced semantic search error: {e}")
            return []
    
    def find_memory_clusters(
        self,
        user_id: str = "default",
        cluster_threshold: float = 0.6,
        min_cluster_size: int = 3
    ) -> Dict[str, List[Dict]]:
        """Findet thematische Cluster in User-Memories"""
        
        try:
            # Get all user memories
            from ..storage.memory_database import MemorySearchFilter  # ✅ KORRIGIERT
            
            search_filter = MemorySearchFilter(
                user_id=user_id,
                limit=500  # Large limit for clustering
            )
            
            all_memories = self.memory_database.search_memories(search_filter)
            
            if len(all_memories) < min_cluster_size:
                return {}
            
            # Simple clustering based on content similarity
            clusters = {}
            clustered_memory_ids = set()
            
            for i, memory in enumerate(all_memories):
                if memory['id'] in clustered_memory_ids:
                    continue
                
                # Find similar memories for this one
                similar_memories = [memory]
                memory_content = memory.get('content', '')
                
                for j, other_memory in enumerate(all_memories[i+1:], i+1):
                    if other_memory['id'] in clustered_memory_ids:
                        continue
                    
                    similarity = self._calculate_content_similarity(
                        memory_content, 
                        other_memory.get('content', '')
                    )
                    
                    if similarity >= cluster_threshold:
                        similar_memories.append(other_memory)
                        clustered_memory_ids.add(other_memory['id'])
                
                # Only create cluster if it meets minimum size
                if len(similar_memories) >= min_cluster_size:
                    # Generate cluster name based on common keywords
                    cluster_name = self._generate_cluster_name(similar_memories)
                    clusters[cluster_name] = similar_memories
                    
                    # Mark all memories in this cluster as clustered
                    for mem in similar_memories:
                        clustered_memory_ids.add(mem['id'])
            
            return clusters
            
        except Exception as e:
            logger.error(f"❌ Memory clustering error: {e}")
            return {}
    
    def temporal_memory_analysis(
        self,
        user_id: str = "default",
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Analysiert Memories über Zeit für Patterns und Trends"""
        
        try:
            # Get memories from time range
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            from ..storage.memory_database import MemorySearchFilter  # ✅ KORRIGIERT
            search_filter = MemorySearchFilter(
                user_id=user_id,
                date_from=cutoff_date.isoformat(),
                limit=1000
            )
            
            memories = self.memory_database.search_memories(search_filter)
            
            if not memories:
                return {'error': 'No memories found in time range'}
            
            # Group by time periods
            time_groups = {
                'by_day': {},
                'by_week': {},
                'by_hour': {},
                'trends': []
            }
            
            # Analyze by different time groupings
            for memory in memories:
                try:
                    created_at = datetime.fromisoformat(memory['created_at'])
                    
                    # Group by day
                    day_key = created_at.strftime('%Y-%m-%d')
                    if day_key not in time_groups['by_day']:
                        time_groups['by_day'][day_key] = []
                    time_groups['by_day'][day_key].append(memory)
                    
                    # Group by week
                    week_key = created_at.strftime('%Y-W%U')
                    if week_key not in time_groups['by_week']:
                        time_groups['by_week'][week_key] = []
                    time_groups['by_week'][week_key].append(memory)
                    
                    # Group by hour
                    hour_key = created_at.hour
                    if hour_key not in time_groups['by_hour']:
                        time_groups['by_hour'][hour_key] = 0
                    time_groups['by_hour'][hour_key] += 1
                    
                except Exception as e:
                    logger.error(f"❌ Date parsing error: {e}")
                    continue
            
            # Calculate trends
            daily_counts = [len(memories) for memories in time_groups['by_day'].values()]
            if len(daily_counts) > 1:
                # Simple trend calculation
                recent_avg = sum(daily_counts[-7:]) / min(7, len(daily_counts))  # Last week
                overall_avg = sum(daily_counts) / len(daily_counts)
                
                trend = "increasing" if recent_avg > overall_avg * 1.1 else \
                       "decreasing" if recent_avg < overall_avg * 0.9 else "stable"
                
                time_groups['trends'].append({
                    'type': 'daily_activity',
                    'trend': trend,
                    'recent_average': recent_avg,
                    'overall_average': overall_avg
                })
            
            # Most active time analysis
            if time_groups['by_hour']:
                most_active_hour = max(time_groups['by_hour'], key=time_groups['by_hour'].get)
                time_groups['most_active_hour'] = {
                    'hour': most_active_hour,
                    'count': time_groups['by_hour'][most_active_hour]
                }
            
            return time_groups
            
        except Exception as e:
            logger.error(f"❌ Temporal analysis error: {e}")
            return {'error': str(e)}
    
    def find_memory_gaps(
        self,
        user_id: str = "default",
        topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Identifiziert Wissenslücken in User-Memories"""
        
        try:
            # Get user memories
            from ..storage.memory_database import MemorySearchFilter  # ✅ KORRIGIERT
            
            search_filter = MemorySearchFilter(
                user_id=user_id,
                query=topic if topic else None,
                limit=500
            )
            
            memories = self.memory_database.search_memories(search_filter)
            
            gaps = []
            
            # Simple gap detection based on common patterns
            knowledge_areas = {
                'personal_preferences': ['favorit', 'mag', 'liebe', 'preference'],
                'work_life': ['arbeit', 'job', 'beruf', 'kollege'],
                'family': ['familie', 'eltern', 'geschwister', 'family'],
                'hobbies': ['hobby', 'freizeit', 'interesse', 'sport'],
                'goals': ['ziel', 'plan', 'vorhaben', 'goal']
            }
            
            # Check coverage for each knowledge area
            for area, keywords in knowledge_areas.items():
                area_memories = []
                for memory in memories:
                    content = memory.get('content', '').lower()
                    if any(keyword in content for keyword in keywords):
                        area_memories.append(memory)
                
                # If coverage is low, it's a potential gap
                if len(area_memories) < 2:  # Threshold for "enough" knowledge
                    gaps.append({
                        'knowledge_area': area,
                        'current_memories': len(area_memories),
                        'gap_severity': 'high' if len(area_memories) == 0 else 'medium',
                        'suggested_questions': self._generate_gap_questions(area),
                        'keywords': keywords
                    })
            
            return gaps
            
        except Exception as e:
            logger.error(f"❌ Memory gap analysis error: {e}")
            return []
    
    # === PRIVATE HELPER METHODS ===
    
    def _detect_search_category(self, query: str) -> str:
        """Erkennt die Kategorie einer Suchanfrage"""
        query_lower = query.lower()
        
        # Score each category
        category_scores = {}
        for category, keywords in self.search_categories.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return highest scoring category or 'general'
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return 'general'
    
    def _expand_query_with_synonyms(self, query: str, category: str) -> str:
        """Erweitert Query mit Synonymen und Kategorie-Keywords"""
        expanded_terms = [query]
        
        # Add category-specific terms
        if category in self.search_categories:
            category_terms = self.search_categories[category][:3]  # Top 3 terms
            expanded_terms.extend(category_terms)
        
        # Add synonyms
        query_words = query.lower().split()
        for word in query_words:
            if word in self.synonyms:
                expanded_terms.extend(self.synonyms[word][:2])  # Top 2 synonyms
        
        return ' '.join(expanded_terms)
    
    def _calculate_enhanced_semantic_score(
        self, 
        query: str, 
        memory: Dict, 
        category: str
    ) -> float:
        """Berechnet erweiterten semantischen Score"""
        
        content = memory.get('content', '').lower()
        query_lower = query.lower()
        
        # Base score from existing relevance
        base_score = memory.get('relevance_score', 0.5)
        
        # Query word matches
        query_words = query_lower.split()
        word_matches = sum(1 for word in query_words if word in content)
        word_score = (word_matches / len(query_words)) * 0.3 if query_words else 0
        
        # Category relevance
        category_keywords = self.search_categories.get(category, [])
        category_matches = sum(1 for keyword in category_keywords if keyword in content)
        category_score = min(category_matches * 0.1, 0.2)
        
        # Memory importance boost
        importance = memory.get('importance', 5)
        importance_score = (importance / 10) * 0.2
        
        # Access frequency boost
        access_count = memory.get('access_count', 0)
        frequency_score = min(access_count * 0.05, 0.2)
        
        # Recency boost
        try:
            created_at = datetime.fromisoformat(memory['created_at'])
            days_old = (datetime.now() - created_at).days
            recency_score = max(0, (30 - days_old) / 30) * 0.1  # Boost for recent memories
        except:
            recency_score = 0
        
        # Emotion boost if emotional query
        emotion_score = 0
        if category == 'emotional' and memory.get('emotion_type'):
            emotion_score = 0.15
        
        total_score = (
            base_score + word_score + category_score + 
            importance_score + frequency_score + recency_score + emotion_score
        )
        
        return min(1.0, total_score)  # Cap at 1.0
    
    def _find_keyword_matches(self, query: str, content: str) -> List[str]:
        """Findet übereinstimmende Keywords zwischen Query und Content"""
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        content_words = set(re.findall(r'\b\w{3,}\b', content.lower()))
        
        matches = list(query_words.intersection(content_words))
        return matches
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Berechnet Ähnlichkeit zwischen zwei Texten"""
        words1 = set(re.findall(r'\b\w{3,}\b', content1.lower()))
        words2 = set(re.findall(r'\b\w{3,}\b', content2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        # Boost similarity if both texts are short and have good overlap
        if len(content1) < 200 and len(content2) < 200 and len(intersection) >= 2:
            jaccard_similarity = min(1.0, jaccard_similarity * 1.3)
        
        return jaccard_similarity
    
    def _generate_cluster_name(self, memories: List[Dict]) -> str:
        """Generiert Namen für Memory-Cluster basierend auf gemeinsamen Keywords"""
        
        # Extract keywords from all memories in cluster
        all_words = []
        for memory in memories:
            content = memory.get('content', '')
            words = re.findall(r'\b\w{4,}\b', content.lower())  # Words with 4+ chars
            all_words.extend(words)
        
        # Find most common words (excluding stop words)
        stop_words = {
            'kira', 'assistant', 'system', 'user', 'that', 'this', 'with', 'have', 'will',
            'from', 'they', 'been', 'said', 'each', 'which', 'their', 'time', 'said'
        }
        
        filtered_words = [word for word in all_words if word not in stop_words]
        
        if filtered_words:
            word_counts = Counter(filtered_words)
            top_words = [word for word, count in word_counts.most_common(3)]
            return f"cluster_{' '.join(top_words)}"
        else:
            return f"cluster_{len(memories)}_memories"
    
    def _generate_gap_questions(self, knowledge_area: str) -> List[str]:
        """Generiert Fragen um Wissenslücken zu schließen"""
        
        gap_questions = {
            'personal_preferences': [
                "Was sind deine Lieblings-Hobbies?",
                "Welche Art von Musik hörst du gerne?",
                "Was ist dein Lieblingsfach oder -interesse?"
            ],
            'work_life': [
                "Erzähl mir von deiner Arbeit",
                "Was machst du beruflich?",
                "Wie ist dein typischer Arbeitsalltag?"
            ],
            'family': [
                "Erzähl mir von deiner Familie",
                "Hast du Geschwister?",
                "Lebst du mit deiner Familie zusammen?"
            ],
            'hobbies': [
                "Welche Hobbies hast du?",
                "Was machst du gerne in deiner Freizeit?",
                "Welchen Sport magst du?"
            ],
            'goals': [
                "Was sind deine Ziele für dieses Jahr?",
                "Woran arbeitest du gerade?",
                "Was möchtest du lernen oder erreichen?"
            ]
        }
        
        return gap_questions.get(knowledge_area, [
            f"Erzähl mir mehr über {knowledge_area}",
            f"Was denkst du über {knowledge_area}?"
        ])

# Export
__all__ = ['MemorySearchEnhancer']