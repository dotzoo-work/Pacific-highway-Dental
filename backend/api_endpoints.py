"""
Additional API endpoints for MongoDB data access
"""
#
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict
from datetime import datetime
from mongodb_manager import get_mongodb_manager

router = APIRouter(prefix="/api", tags=["data"])

@router.get("/stats")
async def get_system_stats():
    """Get system and database statistics"""
    
    try:
        mongodb_manager = get_mongodb_manager()
        db_stats = mongodb_manager.get_database_stats()
        
        return {
            "status": "success",
            "database_stats": db_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@router.get("/session/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get statistics for a specific session"""
    
    try:
        mongodb_manager = get_mongodb_manager()
        stats = mongodb_manager.get_session_stats(session_id)
        
        if not stats:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "status": "success",
            "session_stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session stats: {str(e)}")

@router.get("/logs/daily")
async def get_daily_summary(date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format")):
    """Get daily chat summary"""
    
    try:
        mongodb_manager = get_mongodb_manager()
        summary = mongodb_manager.get_daily_summary(date)
        
        return {
            "status": "success",
            "date": date or datetime.utcnow().strftime('%Y-%m-%d'),
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting daily summary: {str(e)}")

@router.get("/logs/system")
async def get_system_logs(
    limit: int = Query(100, ge=1, le=1000),
    level: Optional[str] = Query(None, description="Log level filter")
):
    """Get recent system logs"""
    
    try:
        mongodb_manager = get_mongodb_manager()
        logs = mongodb_manager.get_recent_logs(limit=limit, level=level)
        
        return {
            "status": "success",
            "logs": logs,
            "count": len(logs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system logs: {str(e)}")

@router.get("/conversation/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    
    try:
        mongodb_manager = get_mongodb_manager()
        conversation = mongodb_manager.get_conversation(session_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "status": "success",
            "session_id": session_id,
            "conversation": conversation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting conversation: {str(e)}")

@router.post("/logs/system")
async def log_system_event(
    level: str,
    message: str,
    metadata: Optional[Dict] = None
):
    """Log a system event"""
    
    try:
        mongodb_manager = get_mongodb_manager()
        mongodb_manager.log_system_event(level, message, metadata)
        
        return {
            "status": "success",
            "message": "Event logged successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging event: {str(e)}")

@router.delete("/cleanup/conversations")
async def cleanup_old_conversations(days: int = Query(30, ge=1, le=365)):
    """Clean up old conversations"""
    
    try:
        mongodb_manager = get_mongodb_manager()
        mongodb_manager.cleanup_expired_conversations(days)
        
        return {
            "status": "success",
            "message": f"Cleaned up conversations older than {days} days"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up conversations: {str(e)}")