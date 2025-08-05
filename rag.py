import os
import json
import asyncio
import logging
import io
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import requests
import pdfplumber
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq

from langchain_core.messages import SystemMessage, HumanMessage
import time
import hashlib
from urllib.parse import urlparse
import magic
import docx2txt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
from pathlib import Path
env_file = Path(".env")
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value

# Configuration
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    BEARER_TOKEN = "dbbdb701cfc45d4041e22a03edbfc65753fe9d7b4b9ba1df4884e864f3bb934d"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_CHUNK_SIZE = 1536  # works well with models like MiniLM and e5
    CHUNK_OVERLAP = 200
    SIMILARITY_THRESHOLD = 0.2
    TOP_K = 11
    PINECONE_INDEX_NAME = "insurance-documents"
    PINECONE_REGION = "us-east-1"
    MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB
    REQUEST_TIMEOUT = 60
    MAX_RETRIES = 3

config = Config()

# Validate configuration
if not config.GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables")
if not config.PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY not found in environment variables")

# Initialize LLM and embeddings with error handling
try:
    llm = ChatGroq(
        api_key=config.GROQ_API_KEY, 
        model="llama3-70b-8192",
        temperature=0.3,  # Slightly higher temperature for more complete responses
        max_tokens=2048,  # Explicitly set max tokens
        max_retries=config.MAX_RETRIES
    )
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    logger.info("LLM and embedding model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    raise

security = HTTPBearer()

# Pydantic Models with validation
class QueryRequest(BaseModel):
    documents: str = Field(..., description="Comma-separated URLs to document blobs", min_length=1)
    questions: List[str] = Field(..., description="List of questions to answer", min_items=1, max_items=50)
    
    @validator('questions')
    def validate_questions(cls, v):
        if not all(question.strip() for question in v):
            raise ValueError("All questions must be non-empty strings")
        return [question.strip() for question in v]
    
    @validator('documents')
    def validate_documents(cls, v):
        urls = [url.strip() for url in v.split(',') if url.strip()]
        if not urls:
            raise ValueError("At least one valid document URL must be provided")
        for url in urls:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL format: {url}")
        return v

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers")
    processing_time: float = Field(..., description="Total processing time in seconds")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_retrieved: int = Field(..., description="Total chunks retrieved for all questions")

# Enhanced Document Processor
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.MAX_CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.document_cache = {}
        self.supported_types = {
            'application/pdf': self._extract_pdf_text,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._extract_docx_text,
            'text/plain': self._extract_text_content,
            'text/html': self._extract_text_content
        }

    def _get_document_hash(self, url: str) -> str:
        """Generate a hash for the document URL for caching"""
        return hashlib.md5(url.encode()).hexdigest()

    def download_document(self, url: str) -> Tuple[bytes, str]:
        """Download document and return content with MIME type"""
        try:
            # Validate URL format more strictly
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid URL format: {url}"
                )
            
            # Check if domain is reachable (basic validation)
            import socket
            try:
                socket.gethostbyname(parsed.netloc.split(':')[0])
            except socket.gaierror:
                raise HTTPException(
                    status_code=400,
                    detail=f"Domain not reachable: {parsed.netloc}"
                )
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            with requests.get(
                url, 
                timeout=config.REQUEST_TIMEOUT, 
                headers=headers,
                stream=True
            ) as response:
                response.raise_for_status()
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > config.MAX_DOCUMENT_SIZE:
                    raise HTTPException(
                        status_code=413, 
                        detail=f"Document too large. Max size: {config.MAX_DOCUMENT_SIZE} bytes"
                    )
                
                content = response.content
                
                # Detect MIME type
                try:
                    mime_type = magic.from_buffer(content[:1024], mime=True)
                except:
                    # Fallback to content-type header or URL extension
                    mime_type = response.headers.get('content-type', '').split(';')[0]
                    if not mime_type:
                        if url.lower().endswith('.pdf'):
                            mime_type = 'application/pdf'
                        elif url.lower().endswith('.docx'):
                            mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                        else:
                            mime_type = 'text/plain'
                
                return content, mime_type
                
        except requests.RequestException as e:
            logger.error(f"Failed to download {url}: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to download document: {str(e)}"
            )
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error downloading document: {str(e)}"
            )

    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            # Convert bytes to file-like object
            pdf_file = io.BytesIO(content)
            
            with pdfplumber.open(pdf_file) as pdf:
                text_parts = []
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(f"\n--- Page {page_num + 1} ---\n{page_text.strip()}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
                        continue
                
                full_text = "\n".join(text_parts)
                
                if not full_text.strip():
                    # Try alternative extraction methods
                    logger.info("Standard extraction failed, trying alternative methods")
                    # You could add OCR here if needed (like pytesseract)
                    return "No readable text content found in PDF"
                
                return full_text.strip()
                
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to extract PDF text: {str(e)}"
            )

    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX content"""
        try:
            docx_file = io.BytesIO(content)
            text = docx2txt.process(docx_file)
            return text.strip() if text else "No text content found in document"
        except Exception as e:
            logger.error(f"DOCX extraction failed: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract DOCX text: {str(e)}"
            )

    def _extract_text_content(self, content: bytes) -> str:
        """Extract text from plain text or HTML content"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    text = content.decode(encoding, errors='ignore')
                    if text.strip():
                        return text.strip()
                except:
                    continue
            
            return "Unable to decode text content"
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            return "Failed to extract text content"

    def process_document(self, url: str) -> List[Document]:
        """Process a document and return chunks"""
        doc_hash = self._get_document_hash(url)
        
        if doc_hash in self.document_cache:
            logger.info(f"Using cached document for {url}")
            return self.document_cache[doc_hash]

        try:
            content, mime_type = self.download_document(url)
            logger.info(f"Downloaded document {url} with MIME type: {mime_type}")
            
            # Extract text based on MIME type
            if mime_type in self.supported_types:
                text = self.supported_types[mime_type](content)
            else:
                logger.warning(f"Unsupported MIME type {mime_type}, treating as plain text")
                text = self._extract_text_content(content)

            if not text or len(text.strip()) < 10:
                raise HTTPException(
                    status_code=400,
                    detail="Document appears to be empty or contains insufficient text content"
                )

            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Filter out very short chunks
            meaningful_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 20]
            
            if not meaningful_chunks:
                raise HTTPException(
                    status_code=400,
                    detail="No meaningful text chunks could be extracted from the document"
                )

            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": url,
                        "chunk_id": i,
                        "mime_type": mime_type,
                        "doc_hash": doc_hash
                    }
                )
                for i, chunk in enumerate(meaningful_chunks)
            ]

            self.document_cache[doc_hash] = documents
            logger.info(f"Processed {len(documents)} chunks for {url}")
            return documents

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing document {url}: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error processing document: {str(e)}"
            )

# Enhanced Pinecone Vector Store
class PineconeVectorStore:
    def __init__(self, api_key: str, index_name: str):
        try:
            self.pc = Pinecone(api_key=api_key)
            self.index_name = index_name
            self.dimension = 384
            
            # Check if index exists, create if not
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=config.PINECONE_REGION)
                )
                # Wait for index to be ready
                time.sleep(10)
            
            self.index = self.pc.Index(index_name)
            self.processed_docs = set()
            logger.info(f"Pinecone vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise

    def document_exists(self, doc_hash: str) -> bool:
        """Check if document is already indexed"""
        return doc_hash in self.processed_docs

    async def add_documents(self, documents: List[Document], batch_size: int = 100):
        """Add documents to the vector store in batches"""
        try:
            doc_hash = documents[0].metadata.get('doc_hash')
            
            if self.document_exists(doc_hash):
                logger.info(f"Document {doc_hash} already indexed")
                return

            # Process in batches to avoid memory issues
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                vectors = []
                
                for doc in batch:
                    try:
                        embedding = embedding_model.encode(doc.page_content).tolist()
                        vector = {
                            "id": f"{doc_hash}_{doc.metadata['chunk_id']}",
                            "values": embedding,
                            "metadata": {
                                "text": doc.page_content[:1000],  # Limit metadata size
                                "source": doc.metadata['source'],
                                "chunk_id": doc.metadata['chunk_id'],
                                "doc_hash": doc_hash
                            }
                        }
                        vectors.append(vector)
                    except Exception as e:
                        logger.error(f"Failed to create embedding for chunk {doc.metadata['chunk_id']}: {str(e)}")
                        continue
                
                if vectors:
                    self.index.upsert(vectors=vectors)
                    logger.info(f"Upserted batch of {len(vectors)} vectors")
            
            self.processed_docs.add(doc_hash)
            logger.info(f"Successfully indexed {len(documents)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            raise

    async def similarity_search(self, query: str, top_k: int = config.TOP_K) -> List[Tuple[Document, float]]:
        """Perform similarity search"""
        try:
            query_embedding = embedding_model.encode(query).tolist()
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            documents_with_scores = []
            for match in results.matches:
                if match.score >= config.SIMILARITY_THRESHOLD:
                    doc = Document(
                        page_content=match.metadata.get("text", ""),
                        metadata=match.metadata
                    )
                    documents_with_scores.append((doc, float(match.score)))
            
            logger.info(f"Retrieved {len(documents_with_scores)} relevant chunks for query")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []

    async def delete_documents(self, doc_hashes: List[str]):
        """Delete documents from the vector store"""
        try:
            for doc_hash in doc_hashes:
                # Delete all vectors for this document
                delete_response = self.index.delete(filter={"doc_hash": {"$eq": doc_hash}})
                logger.info(f"Deleted vectors for document {doc_hash}")
                self.processed_docs.discard(doc_hash)
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")

# Enhanced Insurance Query Processor
class InsuranceQueryEnhancer:
    def __init__(self):
        self.insurance_terms = {
            'premium': ['payment', 'installment', 'fee', 'cost'],
            'coverage': ['benefit', 'protection', 'indemnity', 'compensation'],
            'waiting period': ['qualification period', 'cooling period'],
            'grace period': ['extension period', 'buffer period'],
            'maternity': ['pregnancy', 'childbirth', 'delivery'],
            'pre-existing': ['prior condition', 'existing condition'],
            'deductible': ['excess', 'co-payment'],
            'exclusion': ['limitation', 'restriction'],
            'claim': ['settlement', 'reimbursement'],
            'policy': ['contract', 'agreement', 'plan']
        }

    def expand_query(self, query: str) -> str:
        """Expand query with insurance-specific synonyms"""
        query_lower = query.lower()
        expanded_terms = [query]
        
        for main_term, synonyms in self.insurance_terms.items():
            if main_term in query_lower:
                for synonym in synonyms:
                    expanded_terms.append(query.lower().replace(main_term, synonym))
        
        return ' '.join(expanded_terms)

# FastAPI App
app = FastAPI(
    title="Robust RAG System for Insurance Documents",
    description="Advanced RAG system with comprehensive error handling and document processing",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize components
processor = DocumentProcessor()
vector_store = PineconeVectorStore(config.PINECONE_API_KEY, config.PINECONE_INDEX_NAME)
query_enhancer = InsuranceQueryEnhancer()

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != config.BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# API Endpoints
@app.post("/hackrx/run", response_model=QueryResponse)
async def query_retrieval(
    request: QueryRequest, 
    background_tasks: BackgroundTasks, 
    token: str = Depends(verify_token)
):
    start_time = time.time()
    total_chunks_retrieved = 0
    processed_docs = 0
    
    try:
        doc_urls = [url.strip() for url in request.documents.split(',') if url.strip()]
        logger.info(f"Processing {len(doc_urls)} documents and {len(request.questions)} questions")

        # Process documents with better error handling
        doc_hashes = []
        failed_docs = []
        
        for url in doc_urls:
            try:
                doc_hash = processor._get_document_hash(url)
                
                if not vector_store.document_exists(doc_hash):
                    logger.info(f"Processing new document: {url}")
                    documents = processor.process_document(url)
                    await vector_store.add_documents(documents)
                    processed_docs += 1
                else:
                    logger.info(f"Document already processed: {url}")
                    processed_docs += 1
                
                doc_hashes.append(doc_hash)
                
            except HTTPException as e:
                logger.error(f"HTTP error processing document {url}: {e.detail}")
                failed_docs.append(f"{url}: {e.detail}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing document {url}: {str(e)}")
                failed_docs.append(f"{url}: {str(e)}")
                continue

        # If no documents were successfully processed, return error
        if processed_docs == 0:
            error_msg = "No documents could be processed successfully."
            if failed_docs:
                error_msg += f" Errors: {'; '.join(failed_docs[:3])}"  # Show first 3 errors
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )

        # Process questions
        async def process_question(question: str) -> str:
            nonlocal total_chunks_retrieved
            try:
                expanded_query = query_enhancer.expand_query(question)
                retrieved_docs = await vector_store.similarity_search(expanded_query)
                
                if not retrieved_docs:
                    logger.warning(f"No relevant information found for question: {question}")
                    return "No relevant information found in the documents for this question."

                total_chunks_retrieved += len(retrieved_docs)
                
                # Build context from retrieved documents
                context_parts = []
                for i, (doc, score) in enumerate(retrieved_docs):
                    context_parts.append(
                        f"[Chunk {i+1} - Relevance: {score:.3f}]\n{doc.page_content}"
                    )
                
                context = "\n\n".join(context_parts)
                
                # Enhanced system prompt
                system_prompt = """You are an expert insurance policy analyst with comprehensive knowledge of insurance regulations, particularly Indian insurance policies.

Your expertise includes:
- Policy terms, conditions, and exclusions
- Premium calculations and payment structures
- Claim procedures and settlement processes
- Waiting periods, grace periods, coverage limits, and deductibles
- Pre-existing disease clauses, maternity benefits, and specialized treatments
- Regulatory compliance and policy terminology

Instructions for answering:
1. Provide precise, factual answers based exclusively on the document context provided
2. Include specific amounts, percentages, time periods, and conditions when mentioned
3. Clearly state any conditions, limitations, or exclusions that apply
4. Use proper insurance terminology and maintain professional language
5. If information is not available in the context, explicitly state this
6. When referencing policy sections or clauses, mention them if available
7. Provide comprehensive answers that address all aspects of the question

Format your response clearly and professionally."""

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"""Based on the following document context, please answer the question comprehensively:

CONTEXT:
{context}

QUESTION: {question}

Please provide a detailed, accurate answer based solely on the information in the context above.""")
                ]
                
                response = await llm.ainvoke(messages)
                return response.content
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                return f"An error occurred while processing this question: {str(e)}"

        # Process all questions concurrently
        logger.info("Processing questions concurrently...")
        answers = await asyncio.gather(*[process_question(q) for q in request.questions])
        
        processing_time = time.time() - start_time
        logger.info(f"Completed processing in {processing_time:.2f} seconds")
        
        # Schedule cleanup in background
        background_tasks.add_task(vector_store.delete_documents, doc_hashes)
        
        response_data = QueryResponse(
            answers=answers,
            processing_time=processing_time,
            documents_processed=processed_docs,
            chunks_retrieved=total_chunks_retrieved
        )
        
        # Add warning if some documents failed
        if failed_docs:
            logger.warning(f"Some documents failed to process: {failed_docs}")
        
        return response_data

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error in query retrieval: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# 3. Add a dedicated document validation endpoint
@app.post("/validate-documents")
async def validate_documents(
    documents: str,
    token: str = Depends(verify_token)
):
    """Validate document URLs without processing them"""
    try:
        doc_urls = [url.strip() for url in documents.split(',') if url.strip()]
        results = []
        
        for url in doc_urls:
            try:
                # Basic URL validation
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    results.append({
                        "url": url,
                        "valid": False,
                        "error": "Invalid URL format"
                    })
                    continue
                
                # Test connectivity
                response = requests.head(url, timeout=10, allow_redirects=True)
                
                results.append({
                    "url": url,
                    "valid": response.status_code < 400,
                    "status_code": response.status_code,
                    "content_type": response.headers.get('content-type', 'unknown'),
                    "content_length": response.headers.get('content-length', 'unknown')
                })
                
            except Exception as e:
                results.append({
                    "url": url,
                    "valid": False,
                    "error": str(e)
                })
        
        return {
            "validation_results": results,
            "valid_count": sum(1 for r in results if r.get('valid', False)),
            "total_count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Validation error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test basic functionality
        test_embedding = embedding_model.encode("test")
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "3.0.0",
            "components": {
                "embedding_model": "operational",
                "vector_store": "operational",
                "llm": "operational"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/metrics")
async def get_metrics(token: str = Depends(verify_token)):
    """Get system metrics"""
    return {
        "status": "operational",
        "configuration": {
            "pinecone_index": config.PINECONE_INDEX_NAME,
            "embedding_model": config.EMBEDDING_MODEL,
            "max_chunk_size": config.MAX_CHUNK_SIZE,
            "similarity_threshold": config.SIMILARITY_THRESHOLD,
            "top_k": config.TOP_K
        },
        "version": "3.0.0",
        "features": [
            "multi_format_document_processing",
            "pinecone_vector_database",
            "parallel_question_processing",
            "insurance_domain_optimization",
            "robust_error_handling",
            "document_caching",
            "batch_processing"
        ]
    }

@app.post("/webhook")
async def hackathon_webhook(request: dict):
    """Webhook endpoint for hackathon"""
    logger.info(f"Webhook received: {request}")
    return {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_health": await health_check(),
        "api_endpoints": {
            "main_submission": "/hackrx/run",
            "health_check": "/health",
            "metrics": "/metrics"
        }
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler: {str(exc)}")
    logger.error(traceback.format_exc())
    return HTTPException(
        status_code=500,
        detail="An unexpected error occurred. Please try again later."
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("RAG System starting up...")
    logger.info(f"Configuration loaded: Index={config.PINECONE_INDEX_NAME}, Model={config.EMBEDDING_MODEL}")
    
# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("RAG System shutting down...")

# Run the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        log_level="info",
        access_log=True
    )
