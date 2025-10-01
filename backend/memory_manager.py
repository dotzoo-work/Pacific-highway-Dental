"""
Memory and Context Management System
Implements conversation memory and context tracking for better continuity
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from loguru import logger

class MessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class ConversationMessage:
    content: str
    message_type: MessageType
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['message_type'] = self.message_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationMessage':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['message_type'] = MessageType(data['message_type'])
        return cls(**data)

@dataclass
class PatientProfile:
    session_id: str
    first_interaction: datetime
    last_interaction: datetime
    total_interactions: int
    main_concerns: List[str]
    mentioned_symptoms: List[str]
    treatment_history: List[str]
    preferences: Dict[str, Any]
    risk_factors: List[str]
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['first_interaction'] = self.first_interaction.isoformat()
        data['last_interaction'] = self.last_interaction.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PatientProfile':
        data['first_interaction'] = datetime.fromisoformat(data['first_interaction'])
        data['last_interaction'] = datetime.fromisoformat(data['last_interaction'])
        return cls(**data)

class ConversationMemory:
    """Manages conversation history and context"""
    
    def __init__(self, session_id: str, max_messages: int = 50):
        self.session_id = session_id
        self.max_messages = max_messages
        self.messages: List[ConversationMessage] = []
        self.patient_profile: Optional[PatientProfile] = None
        self.context_summary = ""
        
    def add_message(self, content: str, message_type: MessageType, metadata: Dict = None):
        """Add a new message to conversation history"""
        message = ConversationMessage(
            content=content,
            message_type=message_type,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        
        # Keep only recent messages to manage memory
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        # Update patient profile
        self._update_patient_profile(message)
        
        logger.debug(f"Added {message_type.value} message to conversation {self.session_id}")
    
    def get_recent_context(self, num_messages: int = 10) -> List[Dict]:
        """Get recent conversation context for AI model"""
        recent_messages = self.messages[-num_messages:] if self.messages else []
        
        context = []
        for msg in recent_messages:
            context.append({
                "role": msg.message_type.value,
                "content": msg.content
            })
        
        return context
    
    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation for context"""
        if not self.messages:
            return "No previous conversation history."
        
        user_messages = [msg for msg in self.messages if msg.message_type == MessageType.USER]
        
        if not user_messages:
            return "No user questions in conversation history."
        
        # Extract key topics and concerns
        topics = []
        symptoms = []
        
        for msg in user_messages:
            content_lower = msg.content.lower()
            
            # Extract symptoms
            symptom_keywords = ['pain', 'hurt', 'ache', 'swollen', 'bleeding', 'sensitive']
            for keyword in symptom_keywords:
                if keyword in content_lower:
                    symptoms.append(keyword)
            
            # Extract topics (simplified)
            if len(msg.content) > 20:
                topics.append(msg.content[:100] + "..." if len(msg.content) > 100 else msg.content)
        
        summary_parts = []
        
        if topics:
            summary_parts.append(f"Previous topics discussed: {'; '.join(topics[:3])}")
        
        if symptoms:
            unique_symptoms = list(set(symptoms))
            summary_parts.append(f"Mentioned symptoms: {', '.join(unique_symptoms)}")
        
        if self.patient_profile:
            if self.patient_profile.main_concerns:
                summary_parts.append(f"Main concerns: {', '.join(self.patient_profile.main_concerns[:3])}")
        
        return ". ".join(summary_parts) if summary_parts else "General dental consultation."
    
    def _update_patient_profile(self, message: ConversationMessage):
        """Update patient profile based on new message"""
        if message.message_type != MessageType.USER:
            return
        
        now = datetime.now()
        
        if not self.patient_profile:
            self.patient_profile = PatientProfile(
                session_id=self.session_id,
                first_interaction=now,
                last_interaction=now,
                total_interactions=1,
                main_concerns=[],
                mentioned_symptoms=[],
                treatment_history=[],
                preferences={},
                risk_factors=[]
            )
        else:
            self.patient_profile.last_interaction = now
            self.patient_profile.total_interactions += 1
        
        # Extract information from message
        content_lower = message.content.lower()
        
        # Extract symptoms
        symptom_keywords = {
            'pain': ['pain', 'hurt', 'ache', 'painful'],
            'swelling': ['swollen', 'swelling', 'puffy'],
            'bleeding': ['bleeding', 'blood', 'bleed'],
            'sensitivity': ['sensitive', 'sensitivity', 'sharp'],
            'discoloration': ['yellow', 'brown', 'black', 'stained']
        }
        
        for symptom, keywords in symptom_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                if symptom not in self.patient_profile.mentioned_symptoms:
                    self.patient_profile.mentioned_symptoms.append(symptom)
        
        # Extract concerns (simplified)
        concern_keywords = {
            'cosmetic': ['smile', 'appearance', 'whitening', 'straight'],
            'functional': ['chewing', 'eating', 'bite', 'jaw'],
            'preventive': ['cleaning', 'checkup', 'prevention', 'hygiene'],
            'emergency': ['emergency', 'urgent', 'severe', 'unbearable']
        }
        
        for concern, keywords in concern_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                if concern not in self.patient_profile.main_concerns:
                    self.patient_profile.main_concerns.append(concern)

class MemoryManager:
    """Manages multiple conversation sessions and persistent storage"""
    
    def __init__(self, storage_dir: str = "conversation_memory"):
        self.storage_dir = storage_dir
        self.active_sessions: Dict[str, ConversationMemory] = {}
        self.session_timeout = timedelta(hours=24)  # Sessions expire after 24 hours
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        logger.info(f"Memory Manager initialized with storage: {storage_dir}")
    
    def get_session_id(self, user_identifier: str = None) -> str:
        """Generate or retrieve session ID"""
        if user_identifier:
            # Create consistent session ID for returning users
            return hashlib.md5(user_identifier.encode()).hexdigest()[:16]
        else:
            # Generate random session ID for anonymous users
            return hashlib.md5(str(datetime.now()).encode()).hexdigest()[:16]
    
    def get_conversation(self, session_id: str) -> ConversationMemory:
        """Get or create conversation memory for session"""
        
        # Check if session is already active
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from storage
        conversation = self._load_conversation(session_id)
        
        if conversation:
            # Check if session has expired
            if conversation.patient_profile:
                time_since_last = datetime.now() - conversation.patient_profile.last_interaction
                if time_since_last > self.session_timeout:
                    logger.info(f"Session {session_id} expired, creating new conversation")
                    conversation = ConversationMemory(session_id)
            
            self.active_sessions[session_id] = conversation
            return conversation
        
        # Create new conversation
        conversation = ConversationMemory(session_id)
        self.active_sessions[session_id] = conversation
        
        logger.info(f"Created new conversation for session {session_id}")
        return conversation
    
    def save_conversation(self, session_id: str):
        """Save conversation to persistent storage"""
        if session_id not in self.active_sessions:
            return
        
        conversation = self.active_sessions[session_id]
        file_path = os.path.join(self.storage_dir, f"{session_id}.json")
        
        try:
            data = {
                'session_id': conversation.session_id,
                'messages': [msg.to_dict() for msg in conversation.messages],
                'patient_profile': conversation.patient_profile.to_dict() if conversation.patient_profile else None,
                'context_summary': conversation.context_summary
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved conversation {session_id} to storage")
            
        except Exception as e:
            logger.error(f"Error saving conversation {session_id}: {e}")
    
    def _load_conversation(self, session_id: str) -> Optional[ConversationMemory]:
        """Load conversation from persistent storage"""
        file_path = os.path.join(self.storage_dir, f"{session_id}.json")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversation = ConversationMemory(session_id)
            conversation.context_summary = data.get('context_summary', '')
            
            # Load messages
            for msg_data in data.get('messages', []):
                conversation.messages.append(ConversationMessage.from_dict(msg_data))
            
            # Load patient profile
            if data.get('patient_profile'):
                conversation.patient_profile = PatientProfile.from_dict(data['patient_profile'])
            
            logger.debug(f"Loaded conversation {session_id} from storage")
            return conversation
            
        except Exception as e:
            logger.error(f"Error loading conversation {session_id}: {e}")
            return None
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions from memory and storage"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, conversation in self.active_sessions.items():
            if conversation.patient_profile:
                time_since_last = current_time - conversation.patient_profile.last_interaction
                if time_since_last > self.session_timeout:
                    expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.save_conversation(session_id)  # Save before removing
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up expired session {session_id}")
    
    def get_session_context(self, session_id: str) -> str:
        """Get contextual information for the session"""
        conversation = self.get_conversation(session_id)
        return conversation.get_conversation_summary()

# Example usage
if __name__ == "__main__":
    # Test the memory manager
    memory_manager = MemoryManager()
    
    session_id = memory_manager.get_session_id("test_user")
    conversation = memory_manager.get_conversation(session_id)
    
    # Simulate conversation
    conversation.add_message("My tooth hurts when I drink cold water", MessageType.USER)
    conversation.add_message("I understand that cold sensitivity can be quite uncomfortable...", MessageType.ASSISTANT)
    
    # Save conversation
    memory_manager.save_conversation(session_id)
    
    print(f"Session context: {memory_manager.get_session_context(session_id)}")
#