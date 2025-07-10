"""
Kira Conversation Utilities
Functions for processing and enhancing conversations
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def process_conversation_context(user_input: str, memory_context: Dict[str, Any], 
                               user_emotion: Optional[Dict] = None) -> str:
    """
    Process conversation context and enhance user input with memory and emotion data
    
    Args:
        user_input: Original user input
        memory_context: Context from memory system
        user_emotion: Optional emotion analysis data
        
    Returns:
        Enhanced prompt string
    """
    try:
        # Start with memory enhancement
        enhanced_prompt = enhance_prompt_with_memory(user_input, memory_context)
        
        # Add emotion context if available
        if user_emotion:
            enhanced_prompt = enhance_prompt_with_emotion(enhanced_prompt, user_emotion)
        
        return enhanced_prompt
        
    except Exception as e:
        logger.warning(f"Conversation context processing failed: {e}")
        return user_input


def enhance_prompt_with_memory(user_input: str, memory_context: Dict[str, Any]) -> str:
    """Enhance user input with memory context for better LM Studio responses"""
    try:
        if not memory_context or not memory_context.get('previous_conversations'):
            return user_input

        previous_convs = memory_context.get('previous_conversations', [])
        user_prefs = memory_context.get('user_preferences', {})
        
        context_prompt = f"""
Context from previous conversations:
{chr(10).join(f"- {conv}" for conv in previous_convs[:3])}

User preferences: {user_prefs}

Current message: {user_input}

Please respond naturally, taking into account our conversation history and the user's preferences.
"""
        return context_prompt
        
    except Exception as e:
        logger.warning(f"Prompt enhancement failed: {e}")
        return user_input


def enhance_prompt_with_emotion(prompt: str, user_emotion: Dict[str, Any]) -> str:
    """Enhance prompt with emotional context"""
    try:
        emotion = user_emotion.get('primary_emotion', 'neutral')
        intensity = user_emotion.get('intensity', 0.0)
        
        emotion_guidance = []
        
        if emotion in ['sadness', 'fear', 'anger'] and intensity > 0.6:
            emotion_guidance.append("The user seems to be experiencing strong negative emotions - be especially supportive and empathetic")
        elif emotion in ['joy', 'excitement'] and intensity > 0.6:
            emotion_guidance.append("The user seems happy and excited - match their positive energy appropriately")
        elif emotion == 'confusion':
            emotion_guidance.append("The user seems confused - provide clear, step-by-step explanations")
        elif emotion == 'frustration':
            emotion_guidance.append("The user appears frustrated - be patient and offer helpful solutions")
        
        if emotion_guidance:
            enhanced_prompt = f"""
{prompt}

Emotional context:
{chr(10).join(f"- {guidance}" for guidance in emotion_guidance)}

Please respond naturally while being mindful of the user's emotional state.
"""
            return enhanced_prompt
        
        return prompt
        
    except Exception as e:
        logger.warning(f"Emotion enhancement failed: {e}")
        return prompt


def extract_intent(user_input: str) -> Dict[str, Any]:
    """
    Extract intent from user input using simple pattern matching
    
    Args:
        user_input: User's message
        
    Returns:
        Dictionary with intent analysis
    """
    try:
        text_lower = user_input.lower()
        
        # Question detection
        is_question = '?' in user_input or any(
            word in text_lower for word in ['was', 'wie', 'wer', 'wo', 'wann', 'warum', 'welche']
        )
        
        # Command detection
        is_command = any(
            word in text_lower for word in ['mach', 'erstelle', 'zeig', 'öffne', 'starte', 'stoppe']
        )
        
        # Help request detection
        is_help_request = any(
            word in text_lower for word in ['hilfe', 'help', 'unterstützung', 'kannst du mir helfen']
        )
        
        # Information request
        is_info_request = any(
            word in text_lower for word in ['erkläre', 'was ist', 'information', 'details']
        )
        
        # Emotional expression
        is_emotional = any(
            word in text_lower for word in ['fühle', 'bin traurig', 'freue mich', 'ärgerlich', 'glücklich']
        )
        
        # Determine primary intent
        if is_help_request:
            primary_intent = 'help_request'
        elif is_command:
            primary_intent = 'command'
        elif is_question:
            primary_intent = 'question'
        elif is_info_request:
            primary_intent = 'information_request'
        elif is_emotional:
            primary_intent = 'emotional_expression'
        else:
            primary_intent = 'general_conversation'
        
        # Extract entities (simple keyword extraction)
        entities = extract_simple_entities(user_input)
        
        return {
            'primary_intent': primary_intent,
            'confidence': 0.8 if primary_intent != 'general_conversation' else 0.5,
            'is_question': is_question,
            'is_command': is_command,
            'is_help_request': is_help_request,
            'is_emotional': is_emotional,
            'entities': entities,
            'complexity': calculate_input_complexity(user_input)
        }
        
    except Exception as e:
        logger.error(f"Intent extraction failed: {e}")
        return {
            'primary_intent': 'unknown',
            'confidence': 0.0,
            'error': str(e)
        }


def extract_simple_entities(user_input: str) -> List[Dict[str, Any]]:
    """Extract simple entities from user input"""
    try:
        entities = []
        text_lower = user_input.lower()
        
        # Time entities
        time_indicators = ['heute', 'morgen', 'gestern', 'jetzt', 'später', 'bald']
        for indicator in time_indicators:
            if indicator in text_lower:
                entities.append({
                    'entity': 'time',
                    'value': indicator,
                    'confidence': 0.9
                })
        
        # Technology entities
        tech_indicators = ['computer', 'software', 'programm', 'app', 'website', 'code']
        for indicator in tech_indicators:
            if indicator in text_lower:
                entities.append({
                    'entity': 'technology',
                    'value': indicator,
                    'confidence': 0.8
                })
        
        # Emotion entities
        emotion_indicators = ['glücklich', 'traurig', 'ärgerlich', 'aufgeregt', 'müde', 'gestresst']
        for indicator in emotion_indicators:
            if indicator in text_lower:
                entities.append({
                    'entity': 'emotion',
                    'value': indicator,
                    'confidence': 0.9
                })
        
        return entities
        
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return []


def calculate_input_complexity(user_input: str) -> str:
    """Calculate complexity of user input"""
    try:
        # Length factor
        length_score = len(user_input) / 100.0  # Normalize to ~1.0 for 100 chars
        
        # Word count factor
        word_count = len(user_input.split())
        word_score = word_count / 20.0  # Normalize to ~1.0 for 20 words
        
        # Question complexity (multiple questions, complex structures)
        question_count = user_input.count('?')
        question_score = question_count / 3.0  # Normalize to ~1.0 for 3 questions
        
        # Technical terms
        tech_terms = ['algorithmus', 'datenbank', 'programmierung', 'technologie', 'software']
        tech_score = sum(1 for term in tech_terms if term in user_input.lower()) / 5.0
        
        # Calculate total complexity
        total_complexity = (length_score + word_score + question_score + tech_score) / 4.0
        
        if total_complexity > 0.8:
            return 'high'
        elif total_complexity > 0.5:
            return 'medium'
        else:
            return 'low'
            
    except Exception as e:
        logger.error(f"Complexity calculation failed: {e}")
        return 'medium'


def analyze_topic_category(user_input: str, ai_response: str) -> str:
    """Analyze topic category for memory classification"""
    combined_text = f"{user_input} {ai_response}".lower()
    
    if any(word in combined_text for word in ['lern', 'versteh', 'erklär', 'was ist', 'wie funktioniert']):
        return 'learning'
    elif any(word in combined_text for word in ['gefühl', 'emotion', 'traurig', 'glücklich', 'ärger']):
        return 'emotional'
    elif any(word in combined_text for word in ['problem', 'lösung', 'hilfe', 'schwierigkeit']):
        return 'problem_solving'
    elif any(word in combined_text for word in ['plan', 'zukunft', 'projekt', 'ziel']):
        return 'planning'
    elif any(word in combined_text for word in ['persönlich', 'ich', 'mein', 'privat']):
        return 'personal'
    elif any(word in combined_text for word in ['technical', 'technisch', 'code', 'software', 'computer']):
        return 'technical'
    else:
        return 'general'


def calculate_conversation_importance(user_input: str, ai_response: str) -> int:
    """Calculate conversation importance (1-10)"""
    importance = 5  # Base importance
    
    combined_text = f"{user_input} {ai_response}".lower()
    
    # Length factor
    if len(user_input) > 100 or len(ai_response) > 200:
        importance += 1
    
    # Question complexity
    if '?' in user_input:
        importance += 1
    
    # Important keywords
    if any(word in combined_text for word in ['wichtig', 'dringend', 'remember', 'merken']):
        importance += 2
    
    # Personal content
    if any(word in combined_text for word in ['persönlich', 'privat', 'gefühl', 'emotion']):
        importance += 1
    
    # Learning content
    if any(word in combined_text for word in ['lernen', 'verstehen', 'erklären']):
        importance += 1
    
    return min(10, max(1, importance))


def analyze_emotional_intensity(user_input: str, ai_response: str) -> float:
    """Analyze emotional intensity (0.0-1.0)"""
    combined_text = f"{user_input} {ai_response}".lower()
    
    intensity = 0.0
    
    # Positive emotions
    if any(word in combined_text for word in ['freude', 'glück', 'super', 'toll', 'fantastisch']):
        intensity += 0.3
    
    # Negative emotions  
    if any(word in combined_text for word in ['traurig', 'ärger', 'wut', 'frustration', 'problem']):
        intensity += 0.4
    
    # Excitement indicators
    if '!' in user_input or '!' in ai_response:
        intensity += 0.2
    
    # Caps (shouting)
    if user_input.isupper() and len(user_input) > 5:
        intensity += 0.3
    
    return min(1.0, intensity)


def process_conversation_with_memory(memory_system, user_input: str, ai_response: str, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process conversation with memory system integration
    
    Args:
        memory_system: Memory system instance
        user_input: User's input message
        ai_response: Kira's response
        context: Additional context data
        
    Returns:
        Processing result dictionary
    """
    try:
        from memory.core.memory_types import create_memory, MemoryType
        
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Create enhanced user memory with emotion context
        user_memory_context = {**context, 'role': 'user', 'conversation_id': conversation_id}
        if 'emotion_analysis' in context:
            user_memory_context['user_emotion'] = context['emotion_analysis']
        
        user_memory = create_memory(
            content=f"User: {user_input}",
            memory_type=MemoryType.CONVERSATION,
            importance=context.get('importance', 5),
            context=user_memory_context
        )
        
        # Create enhanced Kira memory with emotion context
        kira_memory_context = {**context, 'role': 'assistant', 'conversation_id': conversation_id}
        if 'kira_emotion' in context:
            kira_memory_context['kira_emotion'] = context['kira_emotion']
        if 'emotional_compatibility' in context:
            kira_memory_context['emotional_compatibility'] = context['emotional_compatibility']
        
        kira_memory = create_memory(
            content=f"Kira: {ai_response}",
            memory_type=MemoryType.CONVERSATION,
            importance=context.get('importance', 5),
            context=kira_memory_context
        )
        
        # Store in memory system
        stored_user = False
        stored_kira = False
        
        if hasattr(memory_system, 'stm') and memory_system.stm:
            stored_user = memory_system.stm.store_memory(user_memory)
            stored_kira = memory_system.stm.store_memory(kira_memory)
        
        return {
            'success': stored_user and stored_kira,
            'conversation_id': conversation_id,
            'storage_result': {
                'storage_location': 'stm_with_emotion',
                'database_stored': False,
                'memory_ids': [user_memory.memory_id, kira_memory.memory_id],
                'conversation_link_stored': True,
                'context_data': {
                    'emotion_analysis': 'emotion_analysis' in context,
                    'personality_applied': context.get('personality_applied', False),
                    'emotional_compatibility': context.get('emotional_compatibility', 0.0)
                }
            },
            'importance_analysis': {
                'calculated_importance': context.get('importance', 5),
                'topic_category': context.get('topic_category', 'general'),
                'emotional_intensity': context.get('emotional_intensity', 0.0)
            }
        }
        
    except Exception as e:
        logger.error(f"Memory conversation processing failed: {e}")
        return {'error': str(e)}


def determine_voice_emotion(text: str) -> str:
    """Determine emotion based on text for voice synthesis"""
    try:
        text_lower = text.lower()

        # Excited indicators
        if any(word in text_lower for word in ['!', 'großartig', 'fantastisch', 'super', 'toll', 'amazing', 'excellent']):
            return 'excited'

        # Empathetic indicators  
        elif any(word in text_lower for word in ['entschuldigung', 'tut mir leid', 'verstehe', 'sorry', 'bedaure']):
            return 'empathetic'

        # Thoughtful indicators
        elif any(word in text_lower for word in ['vielleicht', 'möglicherweise', 'denke', 'überlege', 'interessant']):
            return 'thoughtful'

        # Playful indicators
        elif any(word in text_lower for word in ['😊', '😄', 'spaß', 'witzig', 'lustig', 'haha']):
            return 'playful'

        else:
            return 'calm'  # Default

    except Exception as e:
        logger.warning(f"Emotion determination failed: {e}")
        return 'calm'


def generate_conversation_summary(user_input: str, ai_response: str, context: Dict[str, Any]) -> str:
    """Generate a brief summary of the conversation"""
    try:
        topic_category = analyze_topic_category(user_input, ai_response)
        importance = calculate_conversation_importance(user_input, ai_response)
        
        summary_parts = []
        
        # Add topic
        summary_parts.append(f"Topic: {topic_category}")
        
        # Add importance level
        if importance > 7:
            summary_parts.append("High importance conversation")
        elif importance > 5:
            summary_parts.append("Medium importance conversation")
        
        # Add emotional context if available
        if context.get('emotion_analysis'):
            emotion = context['emotion_analysis'].get('primary_emotion', 'neutral')
            summary_parts.append(f"User emotion: {emotion}")
        
        # Add length indicator
        total_length = len(user_input) + len(ai_response)
        if total_length > 500:
            summary_parts.append("Extended discussion")
        elif total_length > 200:
            summary_parts.append("Detailed exchange")
        
        return " | ".join(summary_parts)
        
    except Exception as e:
        logger.error(f"Conversation summary generation failed: {e}")
        return "Conversation recorded"