"""
Chat Logger for Advanced AI Dental Chatbot
Logs all chat interactions with detailed metadata for analysis
"""
#
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import csv

class ChatLogger:
    """Comprehensive chat logging system"""
    
    def __init__(self, log_dir: str = "chat_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create log files
        self.json_log_file = self.log_dir / "chat_interactions.jsonl"
        self.csv_log_file = self.log_dir / "chat_summary.csv"
        self.detailed_log_file = self.log_dir / "detailed_chat.log"
        
        # Initialize CSV if it doesn't exist
        self._init_csv_log()
    
    def _init_csv_log(self):
        """Initialize CSV log file with headers"""
        if not self.csv_log_file.exists():
            headers = [
                'timestamp', 'session_id', 'user_message', 'ai_response', 
                'agent_type', 'confidence', 'quality_score', 'attempts_used',
                'response_time_ms', 'reasoning_steps_count', 'context_chunks'
            ]
            
            with open(self.csv_log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def log_chat_interaction(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        agent_type: str = "unknown",
        confidence: float = 0.0,
        quality_score: float = 0.0,
        attempts_used: int = 1,
        response_time_ms: int = 0,
        reasoning_steps: list = None,
        context_chunks: int = 0,
        conversation_context: str = "",
        metadata: Dict[str, Any] = None
    ):
        """Log a complete chat interaction with all metadata"""
        
        timestamp = datetime.now().isoformat()
        reasoning_steps = reasoning_steps or []
        metadata = metadata or {}
        
        # Prepare data for logging
        log_data = {
            'timestamp': timestamp,
            'session_id': session_id,
            'user_message': user_message,
            'ai_response': ai_response,
            'agent_type': agent_type,
            'confidence': confidence,
            'quality_score': quality_score,
            'attempts_used': attempts_used,
            'response_time_ms': response_time_ms,
            'reasoning_steps': reasoning_steps,
            'reasoning_steps_count': len(reasoning_steps),
            'context_chunks': context_chunks,
            'conversation_context': conversation_context,
            'metadata': metadata
        }
        
        # Log to JSON file (detailed)
        self._log_to_json(log_data)
        
        # Log to CSV file (summary)
        self._log_to_csv(log_data)
        
        # Log to detailed text file
        self._log_to_detailed_text(log_data)
    
    def _log_to_json(self, log_data: Dict[str, Any]):
        """Log to JSONL file for detailed analysis"""
        try:
            with open(self.json_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error logging to JSON: {e}")
    
    def _log_to_csv(self, log_data: Dict[str, Any]):
        """Log to CSV file for easy analysis"""
        try:
            row = [
                log_data['timestamp'],
                log_data['session_id'],
                log_data['user_message'][:100] + "..." if len(log_data['user_message']) > 100 else log_data['user_message'],
                log_data['ai_response'][:200] + "..." if len(log_data['ai_response']) > 200 else log_data['ai_response'],
                log_data['agent_type'],
                log_data['confidence'],
                log_data['quality_score'],
                log_data['attempts_used'],
                log_data['response_time_ms'],
                log_data['reasoning_steps_count'],
                log_data['context_chunks']
            ]
            
            with open(self.csv_log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"Error logging to CSV: {e}")
    
    def _log_to_detailed_text(self, log_data: Dict[str, Any]):
        """Log to detailed text file for human reading"""
        try:
            with open(self.detailed_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"TIMESTAMP: {log_data['timestamp']}\n")
                f.write(f"SESSION: {log_data['session_id']}\n")
                f.write(f"AGENT: {log_data['agent_type'].upper()}\n")
                f.write(f"CONFIDENCE: {log_data['confidence']:.2f} | QUALITY: {log_data['quality_score']:.1f}\n")
                f.write(f"RESPONSE TIME: {log_data['response_time_ms']}ms | ATTEMPTS: {log_data['attempts_used']}\n")
                f.write(f"CONTEXT CHUNKS: {log_data['context_chunks']}\n")
                f.write(f"\nUSER MESSAGE:\n{log_data['user_message']}\n")
                f.write(f"\nAI RESPONSE:\n{log_data['ai_response']}\n")
                
                if log_data['reasoning_steps']:
                    f.write(f"\nCLINICAL REASONING ({len(log_data['reasoning_steps'])} steps):\n")
                    for i, step in enumerate(log_data['reasoning_steps'], 1):
                        f.write(f"{i}. {step}\n")
                
                if log_data['conversation_context']:
                    f.write(f"\nCONVERSATION CONTEXT:\n{log_data['conversation_context']}\n")
                
                f.write(f"{'='*80}\n")
        except Exception as e:
            print(f"Error logging to detailed text: {e}")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session"""
        stats = {
            'total_interactions': 0,
            'avg_confidence': 0.0,
            'avg_quality': 0.0,
            'avg_response_time': 0.0,
            'agent_types_used': set(),
            'total_reasoning_steps': 0
        }
        
        try:
            if not self.json_log_file.exists():
                return stats
            
            interactions = []
            with open(self.json_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get('session_id') == session_id:
                            interactions.append(data)
                    except json.JSONDecodeError:
                        continue
            
            if interactions:
                stats['total_interactions'] = len(interactions)
                stats['avg_confidence'] = sum(i.get('confidence', 0) for i in interactions) / len(interactions)
                stats['avg_quality'] = sum(i.get('quality_score', 0) for i in interactions) / len(interactions)
                stats['avg_response_time'] = sum(i.get('response_time_ms', 0) for i in interactions) / len(interactions)
                stats['agent_types_used'] = list(set(i.get('agent_type', 'unknown') for i in interactions))
                stats['total_reasoning_steps'] = sum(i.get('reasoning_steps_count', 0) for i in interactions)
        
        except Exception as e:
            print(f"Error getting session stats: {e}")
        
        return stats
    
    def get_daily_summary(self, date: str = None) -> Dict[str, Any]:
        """Get summary for a specific date (YYYY-MM-DD format)"""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        summary = {
            'date': date,
            'total_chats': 0,
            'unique_sessions': set(),
            'agent_usage': {},
            'avg_confidence': 0.0,
            'avg_quality': 0.0,
            'avg_response_time': 0.0
        }
        
        try:
            if not self.json_log_file.exists():
                return summary
            
            daily_interactions = []
            with open(self.json_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get('timestamp', '').startswith(date):
                            daily_interactions.append(data)
                            summary['unique_sessions'].add(data.get('session_id', ''))
                    except json.JSONDecodeError:
                        continue
            
            if daily_interactions:
                summary['total_chats'] = len(daily_interactions)
                summary['unique_sessions'] = len(summary['unique_sessions'])
                summary['avg_confidence'] = sum(i.get('confidence', 0) for i in daily_interactions) / len(daily_interactions)
                summary['avg_quality'] = sum(i.get('quality_score', 0) for i in daily_interactions) / len(daily_interactions)
                summary['avg_response_time'] = sum(i.get('response_time_ms', 0) for i in daily_interactions) / len(daily_interactions)
                
                # Agent usage statistics
                for interaction in daily_interactions:
                    agent = interaction.get('agent_type', 'unknown')
                    summary['agent_usage'][agent] = summary['agent_usage'].get(agent, 0) + 1
        
        except Exception as e:
            print(f"Error getting daily summary: {e}")
        
        return summary

# Global logger instance
chat_logger = ChatLogger()

# Example usage
if __name__ == "__main__":
    # Test logging
    chat_logger.log_chat_interaction(
        session_id="test_session_123",
        user_message="My tooth hurts when I drink cold water",
        ai_response="Cold sensitivity often indicates exposed dentin. I recommend using desensitizing toothpaste and scheduling a check-up.",
        agent_type="diagnostic",
        confidence=0.85,
        quality_score=78.5,
        attempts_used=1,
        response_time_ms=1250,
        reasoning_steps=["Analyzed symptoms", "Considered differential diagnosis", "Recommended treatment"],
        context_chunks=5
    )
    
    print("Test log entry created!")
    print("Session stats:", chat_logger.get_session_stats("test_session_123"))
    print("Daily summary:", chat_logger.get_daily_summary())
