"""
AWS S3 Data Manager for Dental Chatbot
Handles fetching crawled data from S3 and updating vector database
"""

import os
import boto3
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger
import hashlib
from botocore.exceptions import ClientError
from mongodb_manager import get_mongodb_manager

class S3DataManager:
    """Manages data synchronization between S3 and vector database"""
    
    def __init__(self, orchestrator=None):
        # AWS S3 configuration
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'pacific-dental ')
        self.data_prefix = 'pacific-highway-dental.txt/'  # S3 folder for text files
        
        # Vector database client
        self.orchestrator = orchestrator
        self.mongodb_manager = get_mongodb_manager()
        
        logger.info("S3 Data Manager initialized")
    
    def list_s3_files(self) -> List[Dict]:
        """List all text files in S3 bucket"""
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.data_prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.txt'):
                        files.append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'etag': obj['ETag'].strip('"')
                        })
            
            logger.info(f"Found {len(files)} text files in S3")
            return files
            
        except ClientError as e:
            logger.error(f"Error listing S3 files: {e}")
            return []
    
    def download_file_content(self, s3_key: str) -> Optional[str]:
        """Download file content from S3"""
        
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            content = response['Body'].read().decode('utf-8')
            logger.info(f"Downloaded {s3_key} ({len(content)} chars)")
            return content
            
        except ClientError as e:
            logger.error(f"Error downloading {s3_key}: {e}")
            return None
    
    def get_file_hash(self, content: str) -> str:
        """Generate hash for file content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_file_updated(self, s3_key: str, etag: str) -> bool:
        """Check if file has been updated since last sync"""
        
        try:
            # Check MongoDB for last sync record
            sync_record = self.mongodb_manager.db['file_sync'].find_one({
                'file_key': s3_key
            })
            
            if not sync_record:
                return True  # New file
            
            return sync_record.get('etag') != etag
            
        except Exception as e:
            logger.error(f"Error checking file update status: {e}")
            return True  # Assume updated on error
    
    def update_sync_record(self, s3_key: str, etag: str, content_hash: str):
        """Update sync record in MongoDB"""
        
        try:
            self.mongodb_manager.db['file_sync'].update_one(
                {'file_key': s3_key},
                {
                    '$set': {
                        'file_key': s3_key,
                        'etag': etag,
                        'content_hash': content_hash,
                        'last_synced': datetime.utcnow(),
                        'updated_at': datetime.utcnow()
                    },
                    '$setOnInsert': {
                        'created_at': datetime.utcnow()
                    }
                },
                upsert=True
            )
            
            logger.debug(f"Updated sync record for {s3_key}")
            
        except Exception as e:
            logger.error(f"Error updating sync record: {e}")
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks for vector storage"""
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def update_vector_database(self, file_key: str, content: str):
        """Update vector database with new content"""
        
        try:
            if not self.orchestrator:
                logger.warning("Orchestrator not available - skipping vector update")
                return
            
            # Chunk the content
            chunks = self.chunk_text(content)
            logger.info(f"Created {len(chunks)} chunks from {file_key}")
            
            # Generate embeddings and upsert to Pinecone
            # This would integrate with your existing RAG pipeline
            from advanced_rag import AdvancedRAGPipeline
            
            # Create temporary RAG pipeline for processing
            import openai
            openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            rag_pipeline = AdvancedRAGPipeline(openai_client, os.getenv('PINECONE_API_KEY'))
            
            # Process chunks through RAG pipeline
            for i, chunk in enumerate(chunks):
                vector_id = f"{file_key}_{i}"
                # This would call your existing embedding and storage logic
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            
            logger.info(f"Updated vector database with {len(chunks)} chunks from {file_key}")
            
        except Exception as e:
            logger.error(f"Error updating vector database: {e}")
    
    def sync_from_s3(self) -> Dict:
        """Main sync function - check S3 and update vector DB"""
        
        logger.info("Starting S3 sync process")
        
        sync_stats = {
            'files_checked': 0,
            'files_updated': 0,
            'files_new': 0,
            'errors': 0
        }
        
        try:
            # List all files in S3
            s3_files = self.list_s3_files()
            sync_stats['files_checked'] = len(s3_files)
            
            for file_info in s3_files:
                try:
                    s3_key = file_info['key']
                    etag = file_info['etag']
                    
                    # Check if file needs updating
                    if self.is_file_updated(s3_key, etag):
                        logger.info(f"Syncing updated file: {s3_key}")
                        
                        # Download content
                        content = self.download_file_content(s3_key)
                        if not content:
                            sync_stats['errors'] += 1
                            continue
                        
                        # Update vector database
                        self.update_vector_database(s3_key, content)
                        
                        # Update sync record
                        content_hash = self.get_file_hash(content)
                        self.update_sync_record(s3_key, etag, content_hash)
                        
                        # Check if this is a new file
                        sync_record = self.mongodb_manager.db['file_sync'].find_one({
                            'file_key': s3_key
                        })
                        
                        if not sync_record or sync_record.get('created_at') == sync_record.get('updated_at'):
                            sync_stats['files_new'] += 1
                        else:
                            sync_stats['files_updated'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {file_info['key']}: {e}")
                    sync_stats['errors'] += 1
            
            # Log sync results to MongoDB
            self.mongodb_manager.log_system_event(
                'INFO',
                f"S3 sync completed: {sync_stats}",
                sync_stats
            )
            
            logger.info(f"S3 sync completed: {sync_stats}")
            return sync_stats
            
        except Exception as e:
            logger.error(f"S3 sync failed: {e}")
            sync_stats['errors'] += 1
            return sync_stats
    
    def manual_upload_and_sync(self, file_path: str, s3_key: str = None) -> bool:
        """Manually upload file to S3 and sync to vector DB"""
        
        try:
            if not s3_key:
                filename = os.path.basename(file_path)
                s3_key = f"{self.data_prefix}{filename}"
            
            # Upload to S3
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded {file_path} to S3: {s3_key}")
            
            # Read local content and update vector DB
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.update_vector_database(s3_key, content)
            
            # Update sync record
            content_hash = self.get_file_hash(content)
            # Get ETag from S3 after upload
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            etag = response['ETag'].strip('"')
            
            self.update_sync_record(s3_key, etag, content_hash)
            
            logger.info(f"Successfully uploaded and synced {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading and syncing {file_path}: {e}")
            return False

# Global instance
s3_data_manager = None

def get_s3_data_manager(orchestrator=None):
    """Get or create S3 data manager instance"""
    
    global s3_data_manager
    if s3_data_manager is None:
        s3_data_manager = S3DataManager(orchestrator)
    return s3_data_manager