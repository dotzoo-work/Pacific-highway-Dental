"""
Advanced RAG Pipeline with OpenAI Embeddings
Implements sophisticated retrieval and ranking using OpenAI embeddings only
"""

import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import openai
from pinecone import Pinecone
import pinecone
from loguru import logger
import tiktoken
try:
    import boto3
except ImportError:
    boto3 = None
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class RetrievedChunk:
    content: str
    score: float
    chunk_id: str
    metadata: Dict
    relevance_score: float = 0.0
    semantic_score: float = 0.0

class AdvancedRAGPipeline:
    """Enhanced RAG system using OpenAI embeddings for retrieval and ranking"""

    def __init__(self, openai_client, pinecone_api_key: str, index_name: str = "pacific-highway-dental"):
        self.openai_client = openai_client
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        
        # Check if index exists, create if not
        self._ensure_index_exists()
        self.index = self.pc.Index(index_name)
        
        # Load data from S3 if index is empty
        self._load_data_if_empty()

        # Token counter for context management
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        self.max_context_tokens = 5000
        
        # FAQ context
        self.faq_context = self._get_faq_context()

        logger.info("Advanced RAG Pipeline initialized with OpenAI embeddings and FAQ context")
    
    def check_vector_db_status(self):
        """Check vector database status and content"""
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Vector DB Status:")
            logger.info(f"- Total vectors: {stats['total_vector_count']}")
            logger.info(f"- Dimension: {stats.get('dimension', 'Unknown')}")
            
            if stats['total_vector_count'] > 0:
                # Test query to see sample data
                test_results = self.index.query(
                    vector=[0.0] * 1536,  # Dummy vector
                    top_k=3,
                    include_metadata=True
                )
                logger.info(f"Sample data in vector DB:")
                for i, match in enumerate(test_results['matches']):
                    source = match['metadata'].get('source', 'Unknown')
                    text_preview = match['metadata'].get('text', '')[:100]
                    logger.info(f"  {i+1}. Source: {source}, Preview: {text_preview}...")
            
            return stats['total_vector_count']
        except Exception as e:
            logger.error(f"Error checking vector DB status: {e}")
            return 0
    
    def _ensure_index_exists(self):
        """Check if Pinecone index exists, create if not found"""
        try:
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                logger.info(f"Index '{self.index_name}' not found. Creating new index...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI text-embedding-3-small dimension
                    metric="cosine",
                    spec=pinecone.ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Index '{self.index_name}' created successfully!")
            else:
                logger.info(f"Index '{self.index_name}' already exists.")
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}")
            raise
    
    def _load_data_if_empty(self):
        """Load data from S3 if vector database is empty"""
        try:
            # Check if index has data
            stats = self.index.describe_index_stats()
            if stats['total_vector_count'] == 0:
                logger.info("Vector database is empty. Loading data from S3...")
                self.load_s3_data()
            else:
                logger.info(f"Vector database has {stats['total_vector_count']} vectors")
        except Exception as e:
            logger.error(f"Error checking index stats: {e}")
    
    def load_s3_data(self, bucket_name: str = "pacific-dental", key_prefix: str = ""):
        """Load and embed data from S3 bucket"""
        try:
            if boto3 is None:
                logger.warning("boto3 not available. Skipping S3 data loading.")
                return
                
            # Check if AWS credentials are available
            aws_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            if not aws_key or not aws_secret:
                logger.warning("AWS credentials not found. Skipping S3 data loading.")
                return
            
            logger.info(f"Using AWS Access Key: {aws_key[:10]}...")
            
            s3 = boto3.client(
                's3',
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            
            # List objects in S3 bucket
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=key_prefix)
            
            if 'Contents' not in response:
                logger.warning(f"No files found in S3 bucket {bucket_name} with prefix {key_prefix}")
                return
            
            vectors_to_upsert = []
            
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.txt') or key.endswith('.json'):
                    logger.info(f"Processing file: {key}")
                    
                    # Download file content
                    file_obj = s3.get_object(Bucket=bucket_name, Key=key)
                    content = file_obj['Body'].read().decode('utf-8')
                    
                    # Split content into chunks
                    chunks = self._split_text(content, key)
                    
                    # Create embeddings and prepare for upsert
                    for i, chunk in enumerate(chunks):
                        embedding = self._create_embedding(chunk)
                        vector_id = f"{key}_{i}"
                        
                        vectors_to_upsert.append({
                            'id': vector_id,
                            'values': embedding,
                            'metadata': {
                                'text': chunk,
                                'source': key,
                                'chunk_index': i
                            }
                        })
            
            # Upsert vectors in batches
            if vectors_to_upsert:
                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i + batch_size]
                    self.index.upsert(vectors=batch)
                    logger.info(f"Upserted batch {i//batch_size + 1}")
                
                logger.info(f"Successfully loaded {len(vectors_to_upsert)} vectors from S3")
            
        except Exception as e:
            logger.error(f"Error loading data from S3: {e}")
    
    def _split_text(self, text: str, source: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks for embedding"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return []
    
    def enhanced_retrieval(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievedChunk]:
        """
        Enhanced retrieval using OpenAI embeddings with query expansion
        """

        # 1. Primary vector-based retrieval using OpenAI embeddings
        primary_chunks = self._vector_retrieval(query, top_k)

        # 2. Query expansion for better coverage
        expanded_queries = self.query_expansion(query)
        if not isinstance(expanded_queries, list):
            expanded_queries = [query]
        all_chunks = primary_chunks.copy()

        # 3. Retrieve for expanded queries
        if len(expanded_queries) > 1:
            for expanded_query in expanded_queries[1:]:  # Skip original query
                expanded_chunks = self._vector_retrieval(expanded_query, max(1, top_k // 2))
                all_chunks.extend(expanded_chunks)

        # 4. Remove duplicates and re-rank
        final_chunks = self._deduplicate_and_rank(all_chunks, top_k)

        logger.info(f"Retrieved {len(final_chunks)} chunks using enhanced OpenAI approach")
        return final_chunks
    
    def _vector_retrieval(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Traditional vector-based retrieval using OpenAI embeddings"""
        
        try:
            # Get query embedding
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = response.data[0].embedding
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            chunks = []
            for match in results['matches']:
                chunk = RetrievedChunk(
                    content=match['metadata'].get('text', ''),
                    score=match['score'],
                    chunk_id=match['id'],
                    metadata=match['metadata']
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in vector retrieval: {e}")
            return []
    
    def _deduplicate_and_rank(
        self,
        chunks: List[RetrievedChunk],
        top_k: int
    ) -> List[RetrievedChunk]:
        """Remove duplicates and rank chunks by relevance score"""

        # Remove duplicates based on chunk_id
        unique_chunks = {}
        for chunk in chunks:
            if chunk.chunk_id not in unique_chunks:
                unique_chunks[chunk.chunk_id] = chunk
            else:
                # Keep the one with higher score
                if chunk.score > unique_chunks[chunk.chunk_id].score:
                    unique_chunks[chunk.chunk_id] = chunk

        # Sort by score and return top_k
        final_chunks = sorted(
            unique_chunks.values(),
            key=lambda x: x.score,
            reverse=True
        )[:top_k]

        # Update relevance scores
        for chunk in final_chunks:
            chunk.relevance_score = chunk.score
            chunk.semantic_score = chunk.score

        return final_chunks
    
    def _calculate_relevance_boost(self, chunk: RetrievedChunk, query: str) -> float:
        """Calculate relevance boost based on content analysis"""

        content_lower = chunk.content.lower()
        query_lower = query.lower()

        # Boost for exact keyword matches
        query_words = query_lower.split()
        exact_matches = sum(1 for word in query_words if word in content_lower)
        exact_match_boost = exact_matches * 0.1

        # Boost for medical/dental terms
        medical_terms = ['tooth', 'dental', 'pain', 'treatment', 'procedure', 'diagnosis']
        medical_boost = sum(0.05 for term in medical_terms if term in content_lower)

        # Boost for doctor specific content
        doctor_terms = ['meenakshi', 'tomar', 'sweetu', 'patel', 'dds', 'dmd', 'kent', 'pacific highway', 'laser', 'wcli']
        doctor_boost = sum(0.1 for term in doctor_terms if term in content_lower)

        total_boost = exact_match_boost + medical_boost + doctor_boost
        return min(total_boost, 0.5)  # Cap at 0.5
    
    def _get_faq_context(self) -> str:
        """Get FAQ context with template variables replaced"""
        return """
FREQUENTLY ASKED QUESTIONS (FAQ):



Q. Can you schedule an appointment for me? 
A. Unfortunately, I cannot book an appointment directly. Please call our clinic at (253) 529-9434 and our team will be happy to help you with scheduling.




Q. Do you accept Bitcoin? 
A. Pacific Highway Dental does not accept Bitcoin. We accept cash, all major credit cards, and insurance.

Q. Do you offer laughing gas? 
A. Dr. Tomar does not use laughing gas. Instead, she uses single anesthesia to ensure comfort during procedures.

Q. Do you offer silver fillings? 
A. Dr. Tomar does not provide silver (amalgam) fillings. She prefers composite fillings, which are more aesthetically pleasing and often lead to better outcomes.

Q. Do you offer Botox? 
A. Dr. Tomar does not perform Botox treatments. Please contact our clinic for details on the procedures we do offer.

Q. Do you treat toddlers? 
A. Yes, Dr. Tomar treats toddlers. Please contact our clinic for specific details and to schedule an appointment.

Q. Do you have a hygienist? 
A. Pacific Highway Dental does not have a hygienist. Instead, Dr. Tomar is supported by highly qualified dental assistants personally recommended by him, ensuring comprehensive and reliable dental care.

Q. Do you accept Medicare? 
A. Pacific Highway Dental does not participate in Medicare insurance plans. Please contact our clinic to learn about our in-house plans and options.

Q. Do you accept Apple Health? 
A. Pacific Highway Dental does not participate in Apple Health plans. Please contact our clinic to learn about our in-house plans and options.

Q. Do you offer free coffee at your clinic? 
A. We do not offer free coffee, but we do our best to make your visit as comfortable as possible.

Q. Can I use the bathroom at your clinic? 
A. Our facilities are reserved for patients with appointments.

Q. Who are the doctors at Pacific Highway Dental?
A. We have two experienced doctors: Dr. Meenakshi Tomar, DDS and Dr. Sweetu Patel, DMD. Both doctors provide comprehensive dental care with expertise in various dental procedures.

Q. Do you have another dental location? 
A. Yes. Dr. Tomar also practices at Edmonds Bay Dental Clinic. Location - 51 W Dayton Street Suite 301, Edmonds, WA 98020. Please contact the clinic for exact address and hours.

Q: Can I see Dr. Tomar if I live out of state?
A: Yes, absolutely! Dr. Tomar welcomes patients from any location, including those traveling from out of state. Our office offers flexible, travel-friendly scheduling to make your visit as convenient as possible.

Q: Do you treat patients who travel from other cities or states?
A: Yes, we regularly accommodate patients from across the country. There are no geographical restrictions — Dr. Tomar provides comprehensive dental care to all patients, no matter where they’re coming from.
Q. What time is it right now?
A. I am unable to answer that question. I'm a virtual assistant for Dr. Meenakshi Tomar and can only assist with dental and oral health-related inquiries. How can I help you with your dental needs today?

"""
    
    def context_optimization(self, chunks: List[RetrievedChunk], query: str) -> str:
        """Optimize context for token limits and relevance with FAQ integration"""
        
        context_parts = []
        total_tokens = 0
        
        # Add FAQ context first
        faq_tokens = len(self.encoding.encode(self.faq_context))
        if faq_tokens < self.max_context_tokens // 3:  # Use max 1/3 for FAQ
            context_parts.append(self.faq_context + "\n\n")
            total_tokens += faq_tokens
        
        if not chunks:
            return self.faq_context
        
        # Sort chunks by relevance score
        sorted_chunks = sorted(chunks, key=lambda x: x.relevance_score, reverse=True)
        
        # Add query-specific context header
        context_header = f"RELEVANT MEDICAL KNOWLEDGE FOR: {query}\n\n"
        header_tokens = len(self.encoding.encode(context_header))
        total_tokens += header_tokens
        
        for i, chunk in enumerate(sorted_chunks):
            chunk_text = f"CONTEXT {i+1}:\n{chunk.content}\n\n"
            chunk_tokens = len(self.encoding.encode(chunk_text))
            
            # Check if adding this chunk would exceed token limit
            if total_tokens + chunk_tokens > self.max_context_tokens:
                logger.info(f"Context truncated at {i} chunks due to token limit")
                break
            
            context_parts.append(chunk_text)
            total_tokens += chunk_tokens
        
        # Add knowledge base chunks
        for i, chunk in enumerate(sorted_chunks):
            chunk_text = f"CONTEXT {i+1}:\n{chunk.content}\n\n"
            chunk_tokens = len(self.encoding.encode(chunk_text))
            
            # Check if adding this chunk would exceed token limit
            if total_tokens + chunk_tokens > self.max_context_tokens:
                logger.info(f"Context truncated at {i} chunks due to token limit")
                break
            
            context_parts.append(chunk_text)
            total_tokens += chunk_tokens
        
        final_context = "".join(context_parts)
        
        logger.info(f"Optimized context with FAQ: {total_tokens} tokens, {len(sorted_chunks) if chunks else 0} chunks")
        return final_context
    
    def query_expansion(self, original_query: str) -> List[str]:
        """Expand query with related terms for better retrieval"""
        
        expansion_prompt = f"""Given this dental/medical query, generate 3-5 related search terms or phrases that would help find relevant information:

Original query: "{original_query}"

Generate related terms that cover:
1. Medical/dental terminology
2. Symptoms or conditions mentioned
3. Treatment options
4. Related procedures or concepts

Return only the related terms, one per line:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": expansion_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            expanded_terms = response.choices[0].message.content.strip().split('\n')
            expanded_terms = [term.strip() for term in expanded_terms if term.strip()]
            
            logger.info(f"Query expanded with {len(expanded_terms)} additional terms")
            return [original_query] + expanded_terms
            
        except Exception as e:
            logger.error(f"Error in query expansion: {e}")
            return [original_query]
    
    def retrieve_and_rank(
        self,
        query: str,
        use_query_expansion: bool = True,
        top_k: int = 5
    ) -> Tuple[str, List[RetrievedChunk]]:
        """
        Main retrieval method using enhanced OpenAI-based approach
        Returns optimized context and retrieved chunks
        """

        # Use enhanced retrieval with OpenAI embeddings
        chunks = self.enhanced_retrieval(query, top_k * 2)

        # Apply relevance boosting
        for chunk in chunks:
            boost = self._calculate_relevance_boost(chunk, query)
            chunk.relevance_score = chunk.score + boost

        # Re-rank with boosted scores
        final_chunks = sorted(chunks, key=lambda x: x.relevance_score, reverse=True)[:top_k]

        # Optimize context
        optimized_context = self.context_optimization(final_chunks, query)

        return optimized_context, final_chunks

# Example usage and testing
if __name__ == "__main__":
    # This would be used in the main application
    pass
