"""
🔍 INTELLIGENT MEMORY SEARCH ENGINE
Erweiterte Suchfunktionen mit Relevanz-Scoring und Multi-Modal Search
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import re
import json
from collections import defaultdict
import math

from .memory_types import Memory, MemoryType, MemoryImportance

logger = logging.getLogger(__name__)

class MemorySearchEngine:
    """
    🔍 INTELLIGENT MEMORY SEARCH ENGINE
    Erweiterte Suchfunktionen mit AI-Integration
    """
    
    def __init__(self, 
                 stm_system=None, 
                 ltm_system=None,
                 enable_semantic_search: bool = True,
                 enable_temporal_weighting: bool = True):
        """
        Initialize Search Engine
        
        Args:
            stm_system: Short-term memory system
            ltm_system: Long-term memory system
            enable_semantic_search: Enable semantic similarity
            enable_temporal_weighting: Weight results by recency
        """
        self.stm_system = stm_system
        self.ltm_system = ltm_system
        self.enable_semantic_search = enable_semantic_search
        self.enable_temporal_weighting = enable_temporal_weighting
        
        # Search caching
        self._search_cache = {}
        self._cache_expiry = {}
        self.cache_duration = 300  # 5 minutes
        
        # Search analytics
        self.search_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'cached_results': 0,
            'semantic_searches': 0,
            'keyword_searches': 0,
            'temporal_searches': 0
        }
        
        # Keyword extraction patterns
        self.keyword_patterns = {
            'questions': r'\b(was|wie|wann|wo|warum|wer|welche?)\b',
            'emotions': r'\b(freude|trauer|ärger|angst|überraschung|ekel|glück|liebe|hass)\b',
            'actions': r'\b(machen|tun|erstellen|lernen|verstehen|erklären|zeigen)\b',
            'time_references': r'\b(heute|gestern|morgen|letzte woche|nächste woche|vor \d+ tagen)\b',
            'importance': r'\b(wichtig|dringend|kritisch|wesentlich|bedeutend)\b'
        }
        
        logger.info(f"Memory Search Engine initialized")
    
    def search_memories(self, 
                       query: str,
                       search_mode: str = 'comprehensive',
                       memory_types: Optional[List[MemoryType]] = None,
                       importance_min: int = 1,
                       time_range: Optional[Tuple[datetime, datetime]] = None,
                       limit: int = 20,
                       user_id: Optional[str] = None,
                       enable_cache: bool = True) -> Dict[str, Any]:
        """
        ✅ COMPREHENSIVE MEMORY SEARCH
        
        Args:
            query: Search query
            search_mode: 'keyword', 'semantic', 'temporal', 'comprehensive'
            memory_types: Filter by memory types
            importance_min: Minimum importance level
            time_range: (start_date, end_date) tuple
            limit: Maximum results
            user_id: Filter by user
            enable_cache: Use search caching
            
        Returns:
            Detailed search results with relevance scoring
        """
        try:
            search_start = datetime.now()
            self.search_stats['total_searches'] += 1
            
            # Check cache first
            if enable_cache:
                cache_key = self._generate_cache_key(query, search_mode, memory_types, importance_min, limit)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self.search_stats['cached_results'] += 1
                    return cached_result
            
            # Prepare search filters
            search_filters = {
                'memory_types': memory_types or [],
                'importance_min': importance_min,
                'time_range': time_range,
                'user_id': user_id
            }
            
            # Execute search based on mode
            if search_mode == 'keyword':
                search_results = self._keyword_search(query, search_filters, limit)
                self.search_stats['keyword_searches'] += 1
                
            elif search_mode == 'semantic' and self.enable_semantic_search:
                search_results = self._semantic_search(query, search_filters, limit)
                self.search_stats['semantic_searches'] += 1
                
            elif search_mode == 'temporal':
                search_results = self._temporal_search(query, search_filters, limit)
                self.search_stats['temporal_searches'] += 1
                
            elif search_mode == 'comprehensive':
                search_results = self._comprehensive_search(query, search_filters, limit)
                
            else:
                # Fallback to keyword search
                search_results = self._keyword_search(query, search_filters, limit)
                self.search_stats['keyword_searches'] += 1
            
            # Enhance results with metadata
            enhanced_results = self._enhance_search_results(search_results, query, search_mode)
            
            # Performance metrics
            search_duration = (datetime.now() - search_start).total_seconds()
            
            final_result = {
                'success': True,
                'query': query,
                'search_mode': search_mode,
                'results': enhanced_results['results'],
                'result_count': len(enhanced_results['results']),
                'total_possible': enhanced_results['total_scanned'],
                'search_metadata': {
                    'duration_seconds': search_duration,
                    'search_strategy': enhanced_results['strategy_used'],
                    'relevance_algorithm': enhanced_results['relevance_method'],
                    'filters_applied': search_filters,
                    'cache_used': False
                },
                'search_insights': enhanced_results['insights'],
                'timestamp': search_start.isoformat()
            }
            
            # Cache successful results
            if enable_cache and enhanced_results['results']:
                self._cache_result(cache_key, final_result)
            
            if enhanced_results['results']:
                self.search_stats['successful_searches'] += 1
            
            logger.info(f"🔍 Search completed: '{query}' -> {len(enhanced_results['results'])} results in {search_duration:.3f}s")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return {
                'success': False,
                'query': query,
                'error': str(e),
                'results': [],
                'result_count': 0,
                'search_metadata': {
                    'duration_seconds': 0,
                    'error': str(e)
                }
            }
    
    def _comprehensive_search(self, query: str, filters: Dict, limit: int) -> Dict[str, Any]:
        """
        ✅ COMPREHENSIVE SEARCH - Kombiniert alle Suchmethoden
        """
        try:
            all_results = []
            strategies_used = []
            
            # 1. Keyword Search (Basis)
            keyword_results = self._keyword_search(query, filters, limit * 2)
            if keyword_results['results']:
                all_results.extend(keyword_results['results'])
                strategies_used.append('keyword')
            
            # 2. Semantic Search (falls verfügbar)
            if self.enable_semantic_search:
                try:
                    semantic_results = self._semantic_search(query, filters, limit)
                    if semantic_results['results']:
                        all_results.extend(semantic_results['results'])
                        strategies_used.append('semantic')
                except Exception as e:
                    logger.warning(f"Semantic search failed: {e}")
            
            # 3. Temporal Context Search
            temporal_results = self._temporal_search(query, filters, limit)
            if temporal_results['results']:
                all_results.extend(temporal_results['results'])
                strategies_used.append('temporal')
            
            # 4. Pattern-based Search
            pattern_results = self._pattern_search(query, filters, limit)
            if pattern_results['results']:
                all_results.extend(pattern_results['results'])
                strategies_used.append('pattern')
            
            # Deduplicate by memory_id
            unique_results = {}
            for result in all_results:
                memory_id = result['memory'].memory_id
                if memory_id not in unique_results:
                    unique_results[memory_id] = result
                else:
                    # Combine relevance scores
                    existing_score = unique_results[memory_id]['relevance_score']
                    new_score = result['relevance_score']
                    unique_results[memory_id]['relevance_score'] = max(existing_score, new_score)
                    
                    # Combine search methods
                    existing_methods = unique_results[memory_id].get('found_by', [])
                    new_methods = result.get('found_by', [])
                    unique_results[memory_id]['found_by'] = list(set(existing_methods + new_methods))
            
            # Sort by relevance
            final_results = sorted(
                unique_results.values(),
                key=lambda x: x['relevance_score'],
                reverse=True
            )[:limit]
            
            return {
                'results': final_results,
                'strategy_used': strategies_used,
                'total_scanned': len(all_results),
                'total_unique': len(unique_results)
            }
            
        except Exception as e:
            logger.error(f"Comprehensive search failed: {e}")
            return {'results': [], 'strategy_used': ['fallback'], 'total_scanned': 0}
    
    def _keyword_search(self, query: str, filters: Dict, limit: int) -> Dict[str, Any]:
        """
        ✅ KEYWORD-BASED SEARCH mit erweiterten Features
        """
        try:
            all_results = []
            query_lower = query.lower()
            
            # Extract keywords and phrases
            keywords = self._extract_keywords(query)
            phrases = self._extract_phrases(query)
            
            # Search in STM
            if self.stm_system:
                stm_memories = self.stm_system.get_all_memories()
                for memory in stm_memories:
                    if self._matches_filters(memory, filters):
                        relevance = self._calculate_keyword_relevance(memory, query_lower, keywords, phrases)
                        if relevance > 0:
                            all_results.append({
                                'memory': memory,
                                'relevance_score': relevance,
                                'source': 'stm',
                                'found_by': ['keyword'],
                                'matching_keywords': self._find_matching_keywords(memory, keywords),
                                'matching_phrases': self._find_matching_phrases(memory, phrases)
                            })
            
            # Search in LTM
            if self.ltm_system:
                ltm_memories = self.ltm_system.get_all_memories()
                for memory in ltm_memories:
                    if self._matches_filters(memory, filters):
                        relevance = self._calculate_keyword_relevance(memory, query_lower, keywords, phrases)
                        if relevance > 0:
                            all_results.append({
                                'memory': memory,
                                'relevance_score': relevance,
                                'source': 'ltm',
                                'found_by': ['keyword'],
                                'matching_keywords': self._find_matching_keywords(memory, keywords),
                                'matching_phrases': self._find_matching_phrases(memory, phrases)
                            })
            
            # Sort by relevance
            all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return {
                'results': all_results[:limit],
                'total_scanned': len(all_results),
                'keywords_used': keywords,
                'phrases_used': phrases
            }
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return {'results': [], 'total_scanned': 0}
    
    def _semantic_search(self, query: str, filters: Dict, limit: int) -> Dict[str, Any]:
        """
        ✅ SEMANTIC SEARCH mit Similarity Scoring
        """
        try:
            # Placeholder für echte semantic search
            # In real implementation: Use sentence transformers, word embeddings, etc.
            
            # Fallback: Enhanced keyword search with synonyms
            synonyms = self._get_synonyms(query)
            expanded_query = f"{query} {' '.join(synonyms)}"
            
            # Use keyword search with expanded query
            keyword_results = self._keyword_search(expanded_query, filters, limit)
            
            # Boost semantic relevance
            for result in keyword_results['results']:
                result['relevance_score'] *= 1.2  # Semantic boost
                result['found_by'].append('semantic')
            
            return {
                'results': keyword_results['results'],
                'total_scanned': keyword_results['total_scanned'],
                'synonyms_used': synonyms,
                'expanded_query': expanded_query
            }
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {'results': [], 'total_scanned': 0}
    
    def _temporal_search(self, query: str, filters: Dict, limit: int) -> Dict[str, Any]:
        """
        ✅ TEMPORAL SEARCH - Zeit-basierte Suche
        """
        try:
            all_results = []
            current_time = datetime.now()
            
            # Extract time references from query
            time_keywords = self._extract_time_references(query)
            
            # Get all memories from both systems
            all_memories = []
            if self.stm_system:
                all_memories.extend([(m, 'stm') for m in self.stm_system.get_all_memories()])
            if self.ltm_system:
                all_memories.extend([(m, 'ltm') for m in self.ltm_system.get_all_memories()])
            
            for memory, source in all_memories:
                if self._matches_filters(memory, filters):
                    # Calculate temporal relevance
                    temporal_relevance = self._calculate_temporal_relevance(memory, time_keywords, current_time)
                    
                    if temporal_relevance > 0:
                        # Also check basic keyword match
                        keyword_relevance = self._calculate_basic_relevance(memory, query.lower())
                        
                        combined_relevance = (temporal_relevance * 0.7) + (keyword_relevance * 0.3)
                        
                        if combined_relevance > 0.1:  # Minimum threshold
                            all_results.append({
                                'memory': memory,
                                'relevance_score': combined_relevance,
                                'source': source,
                                'found_by': ['temporal'],
                                'temporal_relevance': temporal_relevance,
                                'time_keywords': time_keywords,
                                'memory_age_days': (current_time - memory.created_at).days if memory.created_at else 0
                            })
            
            # Sort by relevance
            all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return {
                'results': all_results[:limit],
                'total_scanned': len(all_results),
                'time_keywords': time_keywords
            }
            
        except Exception as e:
            logger.error(f"Temporal search failed: {e}")
            return {'results': [], 'total_scanned': 0}
    
    def _pattern_search(self, query: str, filters: Dict, limit: int) -> Dict[str, Any]:
        """
        ✅ PATTERN-BASED SEARCH - Mustererkennung
        """
        try:
            all_results = []
            
            # Detect query patterns
            detected_patterns = self._detect_query_patterns(query)
            
            if not detected_patterns:
                return {'results': [], 'total_scanned': 0}
            
            # Get all memories
            all_memories = []
            if self.stm_system:
                all_memories.extend([(m, 'stm') for m in self.stm_system.get_all_memories()])
            if self.ltm_system:
                all_memories.extend([(m, 'ltm') for m in self.ltm_system.get_all_memories()])
            
            for memory, source in all_memories:
                if self._matches_filters(memory, filters):
                    pattern_relevance = self._calculate_pattern_relevance(memory, detected_patterns)
                    
                    if pattern_relevance > 0:
                        all_results.append({
                            'memory': memory,
                            'relevance_score': pattern_relevance,
                            'source': source,
                            'found_by': ['pattern'],
                            'matching_patterns': detected_patterns,
                            'pattern_score': pattern_relevance
                        })
            
            # Sort by relevance
            all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return {
                'results': all_results[:limit],
                'total_scanned': len(all_results),
                'detected_patterns': detected_patterns
            }
            
        except Exception as e:
            logger.error(f"Pattern search failed: {e}")
            return {'results': [], 'total_scanned': 0}
    
    # ✅ HELPER METHODS
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extrahiert wichtige Keywords aus Query"""
        # Remove stop words and extract meaningful terms
        stop_words = {'der', 'die', 'das', 'und', 'oder', 'aber', 'in', 'an', 'auf', 'für', 'mit', 'zu', 'bei', 'von', 'ist', 'war', 'bin', 'hat', 'haben'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    def _extract_phrases(self, query: str) -> List[str]:
        """Extrahiert wichtige Phrasen aus Query"""
        # Find quoted phrases and common patterns
        phrases = []
        
        # Quoted phrases
        quoted = re.findall(r'"([^"]*)"', query)
        phrases.extend(quoted)
        
        # Common multi-word patterns
        patterns = [
            r'\b\w+\s+\w+\s+\w+\b',  # 3-word phrases
            r'\b\w+\s+\w+\b'         # 2-word phrases
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query.lower())
            phrases.extend(matches)
        
        return list(set(phrases))
    
    def _calculate_keyword_relevance(self, memory: Memory, query_lower: str, keywords: List[str], phrases: List[str]) -> float:
        """Berechnet Keyword-basierte Relevanz"""
        try:
            relevance = 0.0
            
            content_lower = memory.content.lower()
            context_str = json.dumps(memory.context).lower() if memory.context else ""
            tags_str = " ".join(memory.tags).lower() if hasattr(memory, 'tags') and memory.tags else ""
            
            searchable_text = f"{content_lower} {context_str} {tags_str}"
            
            # Exact query match (highest relevance)
            if query_lower in searchable_text:
                relevance += 10.0
            
            # Phrase matches
            for phrase in phrases:
                if phrase.lower() in searchable_text:
                    relevance += 5.0
            
            # Individual keyword matches
            for keyword in keywords:
                if keyword in searchable_text:
                    # More relevance for matches in content vs context
                    if keyword in content_lower:
                        relevance += 3.0
                    elif keyword in context_str:
                        relevance += 1.5
                    elif keyword in tags_str:
                        relevance += 2.0
            
            # Word count factor
            word_overlap = len([k for k in keywords if k in searchable_text])
            if keywords:
                relevance += (word_overlap / len(keywords)) * 5.0
            
            # Importance boost
            relevance += memory.importance * 0.5
            
            # Recent memories get slight boost
            if memory.created_at:
                days_old = (datetime.now() - memory.created_at).days
                if days_old < 7:
                    relevance += 1.0
                elif days_old < 30:
                    relevance += 0.5
            
            return relevance
            
        except Exception as e:
            logger.error(f"Relevance calculation failed: {e}")
            return 0.0
    
    def _matches_filters(self, memory: Memory, filters: Dict) -> bool:
        """Prüft ob Memory den Suchfiltern entspricht"""
        try:
            # Memory type filter
            memory_types = filters.get('memory_types', [])
            if memory_types and memory.memory_type not in memory_types:
                return False
            
            # Importance filter
            importance_min = filters.get('importance_min', 1)
            if memory.importance < importance_min:
                return False
            
            # Time range filter
            time_range = filters.get('time_range')
            if time_range and memory.created_at:
                start_date, end_date = time_range
                if not (start_date <= memory.created_at <= end_date):
                    return False
            
            # User ID filter
            user_id = filters.get('user_id')
            if user_id and memory.context.get('user_id') != user_id:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Filter matching failed: {e}")
            return False
    
    def _enhance_search_results(self, search_results: Dict, query: str, search_mode: str) -> Dict[str, Any]:
        """Erweitert Suchergebnisse um Metadata und Insights"""
        try:
            results = search_results.get('results', [])
            
            # Calculate insights
            insights = {
                'total_relevance': sum(r['relevance_score'] for r in results),
                'average_relevance': sum(r['relevance_score'] for r in results) / len(results) if results else 0,
                'source_distribution': {},
                'memory_type_distribution': {},
                'importance_distribution': {},
                'top_keywords': [],
                'search_quality': 'excellent' if results and len(results) >= 5 else 'good' if results else 'poor'
            }
            
            # Analyze source distribution
            for result in results:
                source = result.get('source', 'unknown')
                insights['source_distribution'][source] = insights['source_distribution'].get(source, 0) + 1
            
            # Analyze memory types
            for result in results:
                mem_type = result['memory'].memory_type.value if hasattr(result['memory'].memory_type, 'value') else str(result['memory'].memory_type)
                insights['memory_type_distribution'][mem_type] = insights['memory_type_distribution'].get(mem_type, 0) + 1
            
            # Analyze importance levels
            for result in results:
                importance = result['memory'].importance
                if importance >= 8:
                    level = 'high'
                elif importance >= 5:
                    level = 'medium'
                else:
                    level = 'low'
                insights['importance_distribution'][level] = insights['importance_distribution'].get(level, 0) + 1
            
            # Extract top keywords from results
            all_keywords = []
            for result in results:
                if 'matching_keywords' in result:
                    all_keywords.extend(result['matching_keywords'])
            
            keyword_counts = defaultdict(int)
            for keyword in all_keywords:
                keyword_counts[keyword] += 1
            
            insights['top_keywords'] = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'results': results,
                'strategy_used': search_results.get('strategy_used', [search_mode]),
                'relevance_method': 'multi_factor_scoring',
                'total_scanned': search_results.get('total_scanned', len(results)),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Result enhancement failed: {e}")
            return {
                'results': search_results.get('results', []),
                'strategy_used': [search_mode],
                'relevance_method': 'basic',
                'total_scanned': 0,
                'insights': {'search_quality': 'error', 'error': str(e)}
            }
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Holt Search Engine Statistiken"""
        try:
            total_searches = self.search_stats['total_searches']
            
            return {
                'search_statistics': self.search_stats.copy(),
                'success_rate': (self.search_stats['successful_searches'] / max(1, total_searches)) * 100,
                'cache_hit_rate': (self.search_stats['cached_results'] / max(1, total_searches)) * 100,
                'search_mode_distribution': {
                    'keyword': self.search_stats['keyword_searches'],
                    'semantic': self.search_stats['semantic_searches'],
                    'temporal': self.search_stats['temporal_searches']
                },
                'cache_info': {
                    'cached_entries': len(self._search_cache),
                    'cache_duration_seconds': self.cache_duration
                },
                'features_enabled': {
                    'semantic_search': self.enable_semantic_search,
                    'temporal_weighting': self.enable_temporal_weighting,
                    'result_caching': True,
                    'pattern_recognition': True
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Search statistics failed: {e}")
            return {'error': str(e)}
    
    # ✅ CACHE MANAGEMENT
    
    def _generate_cache_key(self, query: str, search_mode: str, memory_types: Optional[List], importance_min: int, limit: int) -> str:
        """Generiert Cache Key für Search"""
        key_parts = [
            query.lower(),
            search_mode,
            str(sorted([mt.value if hasattr(mt, 'value') else str(mt) for mt in memory_types]) if memory_types else []),
            str(importance_min),
            str(limit)
        ]
        return "|".join(key_parts)
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Holt Cached Search Result"""
        try:
            if cache_key in self._search_cache:
                # Check expiry
                if cache_key in self._cache_expiry:
                    if datetime.now() < self._cache_expiry[cache_key]:
                        result = self._search_cache[cache_key].copy()
                        result['search_metadata']['cache_used'] = True
                        return result
                    else:
                        # Expired
                        del self._search_cache[cache_key]
                        del self._cache_expiry[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            return None
    
    def _cache_result(self, cache_key: str, result: Dict):
        """Cacht Search Result"""
        try:
            self._search_cache[cache_key] = result
            self._cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
            
            # Cleanup old cache entries
            if len(self._search_cache) > 100:  # Max 100 cached entries
                oldest_key = min(self._cache_expiry.keys(), key=lambda k: self._cache_expiry[k])
                del self._search_cache[oldest_key]
                del self._cache_expiry[oldest_key]
                
        except Exception as e:
            logger.error(f"Cache storage failed: {e}")
    
    # ✅ PLACEHOLDER IMPLEMENTATIONS
    
    def _get_synonyms(self, query: str) -> List[str]:
        """Gets synonyms for semantic search"""
        # Placeholder - in real implementation use synonym dictionaries
        synonym_map = {
            'glücklich': ['froh', 'zufrieden', 'fröhlich'],
            'traurig': ['betrübt', 'melancholisch', 'niedergeschlagen'],
            'lernen': ['studieren', 'verstehen', 'begreifen'],
            'problem': ['schwierigkeit', 'herausforderung', 'issue']
        }
        
        synonyms = []
        for word in query.lower().split():
            if word in synonym_map:
                synonyms.extend(synonym_map[word])
        
        return synonyms[:5]  # Limit to 5 synonyms
    
    def _extract_time_references(self, query: str) -> List[str]:
        """Extrahiert Zeit-Referenzen aus Query"""
        time_patterns = {
            'heute': 0,
            'gestern': 1,
            'vorgestern': 2,
            'letzte woche': 7,
            'letzten monat': 30,
            'vor kurzer zeit': 3,
            'kürzlich': 7,
            'neulich': 14
        }
        
        found_times = []
        query_lower = query.lower()
        
        for time_ref in time_patterns:
            if time_ref in query_lower:
                found_times.append(time_ref)
        
        return found_times
    
    def _calculate_temporal_relevance(self, memory: Memory, time_keywords: List[str], current_time: datetime) -> float:
        """Berechnet temporal relevance"""
        if not memory.created_at or not time_keywords:
            return 0.0
        
        memory_age = (current_time - memory.created_at).days
        
        # Map time keywords to relevance
        for time_keyword in time_keywords:
            if time_keyword == 'heute' and memory_age == 0:
                return 10.0
            elif time_keyword == 'gestern' and memory_age == 1:
                return 9.0
            elif time_keyword == 'letzte woche' and memory_age <= 7:
                return 8.0
            elif time_keyword == 'kürzlich' and memory_age <= 7:
                return 7.0
            elif time_keyword == 'neulich' and memory_age <= 14:
                return 6.0
        
        return 0.0
    
    def _calculate_basic_relevance(self, memory: Memory, query_lower: str) -> float:
        """Basic keyword relevance"""
        content_lower = memory.content.lower()
        if query_lower in content_lower:
            return 5.0
        
        # Check individual words
        query_words = query_lower.split()
        matching_words = sum(1 for word in query_words if word in content_lower)
        
        if query_words:
            return (matching_words / len(query_words)) * 3.0
        
        return 0.0
    
    def _detect_query_patterns(self, query: str) -> List[str]:
        """Detektiert Query Patterns"""
        patterns = []
        query_lower = query.lower()
        
        for pattern_name, pattern_regex in self.keyword_patterns.items():
            if re.search(pattern_regex, query_lower):
                patterns.append(pattern_name)
        
        return patterns
    
    def _calculate_pattern_relevance(self, memory: Memory, patterns: List[str]) -> float:
        """Berechnet Pattern-basierte Relevanz"""
        relevance = 0.0
        content_lower = memory.content.lower()
        
        for pattern in patterns:
            if pattern == 'questions' and '?' in memory.content:
                relevance += 3.0
            elif pattern == 'emotions' and any(emotion in content_lower for emotion in ['freude', 'trauer', 'ärger', 'glück']):
                relevance += 4.0
            elif pattern == 'importance' and memory.importance >= 7:
                relevance += 5.0
        
        return relevance
    
    def _find_matching_keywords(self, memory: Memory, keywords: List[str]) -> List[str]:
        """Findet matching keywords in memory"""
        content_lower = memory.content.lower()
        return [kw for kw in keywords if kw in content_lower]
    
    def _find_matching_phrases(self, memory: Memory, phrases: List[str]) -> List[str]:
        """Findet matching phrases in memory"""
        content_lower = memory.content.lower()
        return [phrase for phrase in phrases if phrase.lower() in content_lower]

SearchEngine = MemorySearchEngine

# Export
__all__ = ['MemorySearchEngine', 'SearchEngine']