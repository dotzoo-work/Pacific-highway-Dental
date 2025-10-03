"""
Backend API for Dr. Meenakshi Tomar Dental Chatbot
Minimal FastAPI server for deployment
"""
# hello

import os
import uuid
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
from loguru import logger

# Import our modules
from multi_agent_system import MultiAgentOrchestrator, AgentResponse
from memory_manager import MemoryManager, MessageType
from mongodb_manager import get_mongodb_manager
from api_endpoints import router as api_router
from s3_endpoints import router as s3_router
from s3_data_manager import get_s3_data_manager
from simple_transcript_api import router as transcript_router

# Load environment variables
load_dotenv()

# Configure logging
logger.add("logs/chatbot.log", rotation="1 day", retention="7 days", level="INFO")

# Initialize FastAPI app
app = FastAPI(
    title="Pacific Highway Dental Chatbot API",
    description="AI-powered dental consultation API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.pacifichighwaydental.com"],  # Allow all origins for deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include additional API routes
app.include_router(api_router)
app.include_router(s3_router)
app.include_router(transcript_router)

# Global variables
orchestrator: Optional[MultiAgentOrchestrator] = None
memory_manager: Optional[MemoryManager] = None
openai_client: Optional[openai.OpenAI] = None
mongodb_manager = None
s3_data_manager = None

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global orchestrator, memory_manager, openai_client, mongodb_manager, s3_data_manager
    
    try:
        logger.info("🚀 Starting Dental Chatbot API server...")
        
        # Initialize MongoDB Manager
        mongodb_manager = get_mongodb_manager()
        logger.info("✅ MongoDB Manager initialized")
        
        # Initialize OpenAI client
        openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        logger.info("✅ OpenAI client initialized")
        
        # Initialize Memory Manager
        memory_manager = MemoryManager(storage_dir="conversation_memory")
        logger.info("✅ Memory Manager initialized")
        
        # Initialize Multi-Agent Orchestrator
        orchestrator = MultiAgentOrchestrator(
            openai_client=openai_client,
            pinecone_api_key=os.getenv('PINECONE_API_KEY')
        )
        logger.info("✅ Multi-Agent Orchestrator initialized")
        
        # Initialize S3 Data Manager
        try:
            s3_data_manager = get_s3_data_manager(orchestrator)
            logger.info("✅ S3 Data Manager initialized")
        except Exception as e:
            logger.warning(f"S3 Data Manager initialization failed: {e}")
        
        # Cleanup expired conversations in MongoDB
        mongodb_manager.cleanup_expired_conversations()
        
        logger.info("🎉 API ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    
    import time
    start_time = time.time()
    
    try:
        logger.info(f"📝 Received message: {request.message[:100]}...")
        
        # Get or create session
        if request.session_id:
            session_id = request.session_id
        else:
            session_id = memory_manager.get_session_id(request.user_id)
        
        # Get conversation memory
        conversation = memory_manager.get_conversation(session_id)
        
        # Add user message to memory
        conversation.add_message(request.message, MessageType.USER)
        
        # Get conversation context
        conversation_context = conversation.get_conversation_summary()
        
        # Process with multi-agent system
        agent_response: AgentResponse = orchestrator.process_consultation(request.message, conversation_context)
        
        # Add assistant response to memory
        conversation.add_message(
            agent_response.content, 
            MessageType.ASSISTANT,
            metadata={
                "agent_type": agent_response.agent_type.value,
                "confidence": agent_response.confidence,
                "quality_score": agent_response.quality_score,
                "attempts_used": agent_response.attempts_used
            }
        )
        
        # Save conversation (keeping file backup)
        memory_manager.save_conversation(session_id)

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        logger.info(f"✅ Response generated - Agent: {agent_response.agent_type.value}, "
                   f"Quality: {agent_response.quality_score:.1f}, "
                   f"Time: {response_time_ms}ms")

        # Log interaction to MongoDB
        mongodb_manager.log_chat_interaction(
            session_id=session_id,
            user_message=request.message,
            ai_response=agent_response.content,
            agent_type=agent_response.agent_type.value,
            confidence=agent_response.confidence,
            quality_score=agent_response.quality_score,
            attempts_used=agent_response.attempts_used,
            response_time_ms=response_time_ms,
            reasoning_steps=agent_response.reasoning_steps,
            context_chunks=len(conversation_context.split('\n')) if conversation_context else 0,
            conversation_context=conversation_context
        )
        
        # Save conversation to MongoDB
        conversation_dict = {
            'messages': [{'content': msg.content, 'type': msg.message_type.value, 'timestamp': msg.timestamp.isoformat()} for msg in conversation.messages],
            'patient_profile': conversation.patient_profile.__dict__ if conversation.patient_profile else None
        }
        mongodb_manager.save_conversation(session_id, conversation_dict)

        return ChatResponse(
            response=agent_response.content,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        
        return ChatResponse(
            response="I apologize, but I'm experiencing some technical difficulties. Please try again in a moment, or consider calling our office at (425) 775-5162 for immediate assistance.",
            session_id=session_id if 'session_id' in locals() else "error"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    
    try:
        if not all([orchestrator, memory_manager, openai_client]):
            return {"status": "unhealthy", "message": "Components not initialized"}
        
        # Test OpenAI connection
        test_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        
        # Get MongoDB stats
        db_stats = mongodb_manager.get_database_stats()
        
        return {
            "status": "healthy",
            "components": {
                "orchestrator": "initialized",
                "memory_manager": "initialized",
                "openai_client": "connected",
                "mongodb": "connected",
                "active_sessions": len(memory_manager.active_sessions)
            },
            "database_stats": db_stats
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    
    try:
        if memory_manager:
            for session_id in memory_manager.active_sessions:
                memory_manager.save_conversation(session_id)
            logger.info("💾 All conversations saved")
        
        if mongodb_manager:
            mongodb_manager.close_connection()
            
        logger.info("👋 API shutdown complete")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=False
    )
