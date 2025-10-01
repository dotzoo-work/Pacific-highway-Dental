"""
MongoDB Manager for Dr. Meenakshi Tomar Dental Chatbot
Handles chat_logs, conversation_memory, and logs collections
"""
#
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pymongo import MongoClient
from pymongo.collection import Collection
from loguru import logger
import json

class MongoDBManager:
    """MongoDB manager for chatbot data storage"""
    
    def __init__(self):
        # MongoDB connection
        self.mongo_uri = os.getenv('MONGODB_URI', 'mongodb+srv://username:password@cluster0.mongodb.net/')
        self.db_name = 'chatbot_app'
        self.connected = False
        
        try:
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000
            )
            self.db = self.client[self.db_name]
            
            # Collections
            self.chat_logs = self.db['chat_logs']
            self.conversation_memory = self.db['conversation_memory']
            self.logs = self.db['logs']
            
            # Test connection
            self.client.admin.command('ping')
            self.connected = True
            logger.info("✅ MongoDB connected successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ MongoDB connection failed: {e}")
            logger.info("📝 Running without MongoDB - using fallback storage")
            self.connected = False
    
    # Chat Logs Collection Methods
    def log_chat_interaction(self, session_id: str, user_message: str, ai_response: str, 
                           agent_type: str, confidence: float, quality_score: float,
                           attempts_used: int, response_time_ms: int, reasoning_steps: List[str],
                           context_chunks: int, conversation_context: str):
        """Log chat interaction to MongoDB"""
        
        if not self.connected:
            logger.debug("MongoDB not connected - skipping chat log")
            return
            
        try:
            chat_log = {
                'session_id': session_id,
                'timestamp': datetime.utcnow(),
                'user_message': user_message,
                'ai_response': ai_response,
                'agent_type': agent_type,
                'confidence': confidence,
                'quality_score': quality_score,
                'attempts_used': attempts_used,
                'response_time_ms': response_time_ms,
                'reasoning_steps': reasoning_steps,
                'context_chunks': context_chunks,
                'conversation_context': conversation_context[:500],  # Limit size
                'created_at': datetime.utcnow()
            }
            
            result = self.chat_logs.insert_one(chat_log)
            logger.debug(f"Chat interaction logged: {result.inserted_id}")
            
        except Exception as e:
            logger.error(f"Error logging chat interaction: {e}")
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a specific session"""
        
        try:
            pipeline = [
                {'$match': {'session_id': session_id}},
                {'$group': {
                    '_id': '$session_id',
                    'total_interactions': {'$sum': 1},
                    'avg_response_time': {'$avg': '$response_time_ms'},
                    'avg_quality_score': {'$avg': '$quality_score'},
                    'avg_confidence': {'$avg': '$confidence'},
                    'first_interaction': {'$min': '$timestamp'},
                    'last_interaction': {'$max': '$timestamp'}
                }}
            ]
            
            result = list(self.chat_logs.aggregate(pipeline))
            return result[0] if result else {}
            
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {}
    
    def get_daily_summary(self, date: str = None) -> Dict:
        """Get daily chat summary"""
        
        try:
            if not date:
                target_date = datetime.utcnow().date()
            else:
                target_date = datetime.strptime(date, '%Y-%m-%d').date()
            
            start_date = datetime.combine(target_date, datetime.min.time())
            end_date = start_date + timedelta(days=1)
            
            pipeline = [
                {'$match': {
                    'timestamp': {'$gte': start_date, '$lt': end_date}
                }},
                {'$group': {
                    '_id': None,
                    'total_chats': {'$sum': 1},
                    'unique_sessions': {'$addToSet': '$session_id'},
                    'avg_response_time': {'$avg': '$response_time_ms'},
                    'avg_quality_score': {'$avg': '$quality_score'},
                    'agent_types': {'$push': '$agent_type'}
                }},
                {'$project': {
                    'total_chats': 1,
                    'unique_sessions': {'$size': '$unique_sessions'},
                    'avg_response_time': {'$round': ['$avg_response_time', 2]},
                    'avg_quality_score': {'$round': ['$avg_quality_score', 2]},
                    'agent_types': 1
                }}
            ]
            
            result = list(self.chat_logs.aggregate(pipeline))
            return result[0] if result else {}
            
        except Exception as e:
            logger.error(f"Error getting daily summary: {e}")
            return {}
    
    # Conversation Memory Collection Methods
    def save_conversation(self, session_id: str, conversation_data: Dict):
        """Save conversation to MongoDB"""
        
        if not self.connected:
            logger.debug("MongoDB not connected - skipping conversation save")
            return
            
        try:
            # Upsert conversation
            self.conversation_memory.update_one(
                {'session_id': session_id},
                {
                    '$set': {
                        'session_id': session_id,
                        'conversation_data': conversation_data,
                        'updated_at': datetime.utcnow()
                    },
                    '$setOnInsert': {
                        'created_at': datetime.utcnow()
                    }
                },
                upsert=True
            )
            
            logger.debug(f"Conversation saved for session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    def get_conversation(self, session_id: str) -> Optional[Dict]:
        """Get conversation from MongoDB"""
        
        if not self.connected:
            return None
            
        try:
            result = self.conversation_memory.find_one({'session_id': session_id})
            return result['conversation_data'] if result else None
            
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return None
    
    def cleanup_expired_conversations(self, days: int = 30):
        """Clean up old conversations"""
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            result = self.conversation_memory.delete_many({
                'updated_at': {'$lt': cutoff_date}
            })
            
            logger.info(f"Cleaned up {result.deleted_count} expired conversations")
            
            # Log cleanup event
            self.log_system_event('INFO', f'Cleaned up {result.deleted_count} expired conversations', {
                'deleted_count': result.deleted_count,
                'cutoff_date': cutoff_date.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error cleaning up conversations: {e}")
    
    # System Logs Collection Methods
    def log_system_event(self, level: str, message: str, metadata: Dict = None):
        """Log system events to MongoDB"""
        
        if not self.connected:
            logger.debug("MongoDB not connected - skipping system log")
            return
            
        try:
            log_doc = {
                'level': level,
                'message': message,
                'metadata': metadata or {},
                'timestamp': datetime.utcnow(),
                'created_at': datetime.utcnow()
            }
            
            result = self.logs.insert_one(log_doc)
            logger.debug(f"System event logged: {result.inserted_id}")
            
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
    
    def get_recent_logs(self, limit: int = 100, level: str = None) -> List[Dict]:
        """Get recent system logs"""
        
        if not self.connected:
            return []
            
        try:
            query = {}
            if level:
                query['level'] = level
            
            logs = list(self.logs.find(query)
                       .sort('timestamp', -1)
                       .limit(limit))
            
            # Convert ObjectId to string for JSON serialization
            for log in logs:
                log['_id'] = str(log['_id'])
            
            return logs
            
        except Exception as e:
            logger.error(f"Error getting recent logs: {e}")
            return []
    
    # Utility Methods
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        
        try:
            stats = {
                'chat_logs_count': self.chat_logs.count_documents({}),
                'conversation_memory_count': self.conversation_memory.count_documents({}),
                'system_logs_count': self.logs.count_documents({}),
                'database_name': self.db_name,
                'collections': self.db.list_collection_names()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def close_connection(self):
        """Close MongoDB connection"""
        
        try:
            self.client.close()
            logger.info("MongoDB connection closed")
            
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")

# Global MongoDB manager instance
mongodb_manager = None

def get_mongodb_manager() -> MongoDBManager:
    """Get or create MongoDB manager instance"""
    
    global mongodb_manager
    if mongodb_manager is None:
        mongodb_manager = MongoDBManager()
    return mongodb_manager