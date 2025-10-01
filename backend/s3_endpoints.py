"""
S3 Data Management API Endpoints
"""
#
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from typing import Dict
from s3_data_manager import get_s3_data_manager
from loguru import logger
import tempfile
import os

router = APIRouter(prefix="/api/s3", tags=["s3-data"])

@router.post("/sync")
async def trigger_s3_sync(background_tasks: BackgroundTasks):
    """Trigger S3 sync process"""
    
    try:
        s3_manager = get_s3_data_manager()
        
        # Run sync in background
        background_tasks.add_task(s3_manager.sync_from_s3)
        
        return {
            "status": "success",
            "message": "S3 sync started in background"
        }
        
    except Exception as e:
        logger.error(f"Error triggering S3 sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sync")
async def run_s3_sync():
    """Run S3 sync and return results"""
    
    try:
        s3_manager = get_s3_data_manager()
        sync_stats = s3_manager.sync_from_s3()
        
        return {
            "status": "success",
            "sync_stats": sync_stats
        }
        
    except Exception as e:
        logger.error(f"Error running S3 sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files")
async def list_s3_files():
    """List all files in S3 bucket"""
    
    try:
        s3_manager = get_s3_data_manager()
        files = s3_manager.list_s3_files()
        
        return {
            "status": "success",
            "files": files,
            "count": len(files)
        }
        
    except Exception as e:
        logger.error(f"Error listing S3 files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_file_to_s3(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload new text file to S3 and sync to vector DB"""
    
    try:
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Only .txt files are allowed")
        
        s3_manager = get_s3_data_manager()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Upload to S3 and sync in background
        s3_key = f"crawled_data/{file.filename}"
        
        def upload_and_sync():
            try:
                success = s3_manager.manual_upload_and_sync(temp_file_path, s3_key)
                os.unlink(temp_file_path)  # Clean up temp file
                if success:
                    logger.info(f"Successfully uploaded and synced {file.filename}")
                else:
                    logger.error(f"Failed to upload and sync {file.filename}")
            except Exception as e:
                logger.error(f"Background upload error: {e}")
                os.unlink(temp_file_path)  # Clean up on error
        
        background_tasks.add_task(upload_and_sync)
        
        return {
            "status": "success",
            "message": f"File {file.filename} upload and sync started",
            "s3_key": s3_key
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sync-status")
async def get_sync_status():
    """Get sync status and statistics"""
    
    try:
        from mongodb_manager import get_mongodb_manager
        mongodb_manager = get_mongodb_manager()
        
        # Get recent sync records
        recent_syncs = list(mongodb_manager.db['file_sync'].find(
            {},
            {'_id': 0}
        ).sort('last_synced', -1).limit(10))
        
        # Get sync event logs
        sync_logs = mongodb_manager.get_recent_logs(limit=5, level='INFO')
        sync_events = [log for log in sync_logs if 'sync' in log.get('message', '').lower()]
        
        return {
            "status": "success",
            "recent_syncs": recent_syncs,
            "recent_sync_events": sync_events
        }
        
    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        raise HTTPException(status_code=500, detail=str(e))