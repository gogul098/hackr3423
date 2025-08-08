import os
import asyncio
import json
import logging
import httpx
import tenacity
import hashlib
import time
import uuid
import sys
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Optional, Tuple, Any
from email import policy
from email.parser import BytesParser
import aiofiles

import fitz # PyMuPDF
import docx
import spacy
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from sentence_transformers import SentenceTransformer
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import faiss
from cachetools import TTLCache
import redis.asyncio as redis
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from tenacity import Retrying, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception
from redis.exceptions import ConnectionError

# --- 1. CONFIGURATION AND GLOBAL SETUP ---
# --- Structured Logging ---
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "request_id": "%(request_id)s", "user_id": "%(user_id)s"}')
logger = logging.getLogger(__name__)

# Dedicated audit logger
audit_logger = logging.getLogger('audit')
audit_file_handler = logging.FileHandler('audit.log')
audit_file_handler.setFormatter(logging.Formatter('{"time": "%(asctime)s", "user_id": "%(user_id)s", "request_id": "%(request_id)s", "action": "%(message)s"}'))
audit_logger.addHandler(audit_file_handler)
audit_logger.setLevel(logging.INFO)

# Pydantic models for request/response bodies
class Question(BaseModel):
    question: str

class AnswerRequest(BaseModel):
    document_url: HttpUrl
    questions: List[Question]

class AnswerDetail(BaseModel):
    question: str
    answer: str
    confidence_score: float
    rationale: str
    provenance_chunks: List[str]

class AnswerResponse(BaseModel):
    answers: List[AnswerDetail]
    message: str

# Cache directory for documents, indices, and intermediate results
CACHE_DIR = Path("cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Define models for processing
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
QA_EXTRACTIVE_MODEL_NAME = "deepset/roberta-base-squad2"
GENERATIVE_MODEL_NAME = "google/flan-t5-large"
SUMMARIZATION_MODEL_NAME = "google/flan-t5-small"

# API Key Management (placeholder for a real database)
API_KEYS_DB = {
    "super_secret_key_1": {"user_id": "user-1", "scopes": ["read", "write"]},
    "another_secret_key_2": {"user_id": "user-2", "scopes": ["read"]},
}

# Retrieval and answering parameters
CHUNK_SIZE_CHARS = 600
TOP_K_CHUNKS = 5
MAX_PROMPT_TOKENS = 1024
DOC_VERSION = "v3"  # Incrementing the version for a new document processing pipeline

# Redis configuration for distributed caching and rate limiting
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

# Rate limiting settings using a Lua script for atomicity
RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local interval = tonumber(ARGV[2])
local current = tonumber(redis.call('get', key) or "0")
if current == 0 then
    redis.call('incr', key)
    redis.call('expire', key, interval)
    return 1
else
    if current < limit then
        redis.call('incr', key)
        return 1
    else
        return 0
    end
end
"""

# Concurrency control for models using semaphores
MODEL_CONCURRENCY_LIMIT = 5
embedding_semaphore = asyncio.Semaphore(MODEL_CONCURRENCY_LIMIT)
extractive_qa_semaphore = asyncio.Semaphore(MODEL_CONCURRENCY_LIMIT)
generative_semaphore = asyncio.Semaphore(MODEL_CONCURRENCY_LIMIT)
summarization_semaphore = asyncio.Semaphore(MODEL_CONCURRENCY_LIMIT)

# --- Custom Prometheus Metrics ---
documents_processed_counter = Counter("documents_processed_total", "Total number of documents successfully processed.")
questions_answered_counter = Counter("questions_answered_total", "Total number of questions answered.")
embedding_tokens_counter = Counter("embedding_tokens_total", "Total number of tokens embedded.")
generative_tokens_counter = Counter("generative_tokens_total", "Total number of tokens generated.")
processing_latency_seconds = Histogram("document_processing_latency_seconds", "Latency of document processing and indexing.")
qa_latency_seconds = Histogram("question_answering_latency_seconds", "Latency of the entire question answering pipeline.")

# Define FastAPI app
app = FastAPI(title="Document QA API", description="A robust and scalable API for question answering on documents.")

# In-memory status tracking for documents being processed
processing_status = TTLCache(maxsize=100, ttl=3600)

# --- 2. FASTAPI EVENT HANDLERS ---
@app.on_event("startup")
async def startup_event():
    """Run on application startup to load models and connect to Redis."""
    try:
        # Load all models on startup
        get_spacy_model()
        get_embedding_model()
        get_extractive_qa_pipeline()
        get_generative_llm()
        get_summarization_pipeline()

        # Connect to Redis
        app.state.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await app.state.redis_client.ping()
        app.state.rate_limit_script_sha = await app.state.redis_client.script_load(RATE_LIMIT_SCRIPT)
        logger.info("Successfully connected to Redis and loaded Lua script.", extra={"request_id": "system", "user_id": "system"})
    except Exception as e:
        logger.error(f"Failed to initialize application components: {e}", extra={"request_id": "system", "user_id": "system"})
        sys.exit(1)

    # Setup Prometheus metrics
    Instrumentator().instrument(app).expose(app)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    if app.state.redis_client:
        await app.state.redis_client.close()
        logger.info("Redis connection closed.", extra={"request_id": "system", "user_id": "system"})

# --- 3. MIDDLEWARE AND DEPENDENCIES ---
@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """Adds a unique request ID and other context for structured logging."""
    request.state.request_id = str(uuid.uuid4())
    request.state.user_id = "anonymous"
    
    # Audit log the start of the request
    audit_logger.info("Request received", extra={"request_id": request.state.request_id, "user_id": request.state.user_id})

    response = await call_next(request)

    # Audit log the end of the request
    audit_logger.info("Request completed", extra={"request_id": request.state.request_id, "user_id": request.state.user_id})
    return response

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Atomic Redis-backed rate limiting using a Lua script."""
    try:
        ip_address = request.client.host
        key = f"rate_limit:{ip_address}"
        limit = 10
        interval = 60
        
        can_request = await app.state.redis_client.evalsha(
            app.state.rate_limit_script_sha, 1, key, limit, interval
        )

        if not can_request:
            logger.warning(f"Rate limit exceeded for IP: {ip_address}", extra={"request_id": request.state.request_id, "user_id": request.state.user_id})
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
    except ConnectionError as e:
        logger.error(f"Redis connection failed for rate limiting: {e}", extra={"request_id": request.state.request_id, "user_id": request.state.user_id})
        pass
    
    response = await call_next(request)
    return response

class CustomHTTPXClient:
    """HTTPX client with a circuit breaker and backoff strategy."""
    _retryer = Retrying(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        reraise=True,
    )
    _client = httpx.AsyncClient(timeout=30)

    async def get_and_stream_to_file(self, url: str, file_path: Path):
        """Streams content from a URL to a file with retries."""
        try:
            async with self._retryer:
                async with self._client.stream("GET", url, follow_redirects=True) as response:
                    response.raise_for_status()
                    async with aiofiles.open(file_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            await f.write(chunk)
        except Exception as e:
            logger.error(f"Failed to download URL {url}: {e}", extra={"request_id": "system", "user_id": "system"})
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"External document service failed: {e}")

@lru_cache(maxsize=None)
def get_httpx_client():
    return CustomHTTPXClient()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    """Multi-user API key management and scoped authorization."""
    key_info = API_KEYS_DB.get(credentials.credentials)
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return key_info

# --- 4. MODEL LOADING AND CACHING ---
@lru_cache(maxsize=None)
def get_spacy_model():
    """Load the spaCy model and cache it."""
    logger.info("Loading spaCy model: en_core_web_sm", extra={"request_id": "system", "user_id": "system"})
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.error("spaCy model 'en_core_web_sm' not found. Please install it with 'python -m spacy download en_core_web_sm'.", extra={"request_id": "system", "user_id": "system"})
        raise

@lru_cache(maxsize=None)
def get_embedding_model():
    """Load the SentenceTransformer model and cache it."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}", extra={"request_id": "system", "user_id": "system"})
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@lru_cache(maxsize=None)
def get_extractive_qa_pipeline():
    """Load the extractive QA pipeline and cache it."""
    logger.info(f"Loading extractive QA pipeline: {QA_EXTRACTIVE_MODEL_NAME}", extra={"request_id": "system", "user_id": "system"})
    return pipeline("question-answering", model=QA_EXTRACTIVE_MODEL_NAME)

@lru_cache(maxsize=None)
def get_generative_llm():
    """Load the generative T5 model and tokenizer and cache it."""
    logger.info(f"Loading generative model: {GENERATIVE_MODEL_NAME}", extra={"request_id": "system", "user_id": "system"})
    tokenizer = T5Tokenizer.from_pretrained(GENERATIVE_MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(GENERATIVE_MODEL_NAME)
    return model, tokenizer

@lru_cache(maxsize=None)
def get_summarization_pipeline():
    """Load the summarization pipeline and cache it."""
    logger.info(f"Loading summarization pipeline: {SUMMARIZATION_MODEL_NAME}", extra={"request_id": "system", "user_id": "system"})
    return pipeline("summarization", model=SUMMARIZATION_MODEL_NAME)

# --- 5. DOCUMENT PROCESSING UTILITIES ---
def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise
    return text

def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        raise
    return text

def extract_text_from_email(file_path: Path) -> str:
    """Extract text from an EML file."""
    try:
        with open(file_path, 'rb') as fp:
            msg = BytesParser(policy=policy.default).parse(fp)
        
        email_body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition'))
                
                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    email_body += part.get_payload(decode=True).decode()
                    break
        else:
            email_body = msg.get_payload(decode=True).decode()

        email_header = f"Subject: {msg['subject']}\nFrom: {msg['from']}\nTo: {msg['to']}\n\n"
        return email_header + email_body
    except Exception as e:
        logger.error(f"Error extracting text from email: {e}")
        raise

def chunk_text_by_sentences_spacy(text: str) -> List[str]:
    """Intelligently chunk text by sentences using spaCy."""
    nlp = get_spacy_model()
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > CHUNK_SIZE_CHARS:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

async def process_document_and_build_index(doc_url: str, request_id: str, user_id: str):
    """
    Handles the entire document processing pipeline, persisting data to Redis.
    """
    start_time = time.time()
    cache_key = f"{DOC_VERSION}:{hashlib.sha256(doc_url.encode('utf-8')).hexdigest()}"
    redis_client = app.state.redis_client
    
    try:
        if await redis_client.exists(cache_key):
            logger.info(f"Loading cached data from Redis for {doc_url}", extra={"request_id": request_id, "user_id": user_id})
            processing_status[doc_url] = "completed"
            return
        
        logger.info(f"No cache found for {doc_url}. Starting fresh processing.", extra={"request_id": request_id, "user_id": user_id})
        processing_status[doc_url] = "processing"
        
        file_ext = os.path.splitext(doc_url)[1].lower()
        doc_file_path = CACHE_DIR / f"doc_{cache_key}{file_ext}"
        faiss_index_path = CACHE_DIR / f"{cache_key}.faiss"
        
        httpx_client = get_httpx_client()
        await httpx_client.get_and_stream_to_file(doc_url, doc_file_path)

        if file_ext == '.pdf':
            text = await asyncio.to_thread(extract_text_from_pdf, doc_file_path)
        elif file_ext == '.docx':
            text = await asyncio.to_thread(extract_text_from_docx, doc_file_path)
        elif file_ext in ['.eml', '.msg']:
            text = await asyncio.to_thread(extract_text_from_email, doc_file_path)
        else:
            raise ValueError("Unsupported document type.")

        chunks = await asyncio.to_thread(chunk_text_by_sentences_spacy, text)
        
        async with embedding_semaphore:
            embedding_model = get_embedding_model()
            embeddings = await asyncio.to_thread(embedding_model.encode, chunks, show_progress_bar=False)
            embeddings = np.array(embeddings).astype('float32')
            
        faiss.normalize_L2(embeddings)
        faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss_index.add(embeddings)
        faiss.write_index(faiss_index, str(faiss_index_path))

        metadata = {"chunks": chunks, "url": doc_url, "faiss_path": str(faiss_index_path)}
        await redis_client.set(cache_key, json.dumps(metadata))

        documents_processed_counter.inc()
        processing_latency_seconds.observe(time.time() - start_time)
        processing_status[doc_url] = "completed"
        logger.info(f"Successfully processed and cached document for {doc_url}", extra={"request_id": request_id, "user_id": user_id})
    except Exception as e:
        processing_status[doc_url] = "failed"
        logger.error(f"Failed to process document {doc_url}: {e}", extra={"request_id": request_id, "user_id": user_id})
        raise

# --- 6. RETRIEVAL AND ANSWERING LOGIC ---
async def semantic_expand_query(question: str, request_id: str, user_id: str) -> List[str]:
    """Expands a query using a generative model for semantic synonyms."""
    async with generative_semaphore:
        try:
            generative_model, generative_tokenizer = get_generative_llm()
            prompt = f"Provide a list of 3 synonyms or related phrases for the following query. Format the output as a comma-separated list:\n\nQuery: {question}\n\nList:"
            
            inputs = generative_tokenizer(prompt, return_tensors="pt", max_length=100, truncation=True)
            output = await asyncio.to_thread(
                generative_model.generate,
                inputs['input_ids'],
                max_length=50,
                num_return_sequences=1,
                early_stopping=True
            )
            expanded_text = generative_tokenizer.decode(output[0], skip_special_tokens=True)
            expanded_queries = [q.strip() for q in expanded_text.split(',') if q.strip()]
            return list(set(expanded_queries + [question]))
        except Exception as e:
            logger.error(f"Failed to perform semantic query expansion: {e}", extra={"request_id": request_id, "user_id": user_id})
            return [question]

async def process_question_in_background(request_id: str, user_id: str, doc_url: str, question: str):
    """
    An async orchestrator for a single question, using semaphores for model inference.
    """
    start_time = time.time()
    try:
        redis_client = app.state.redis_client
        doc_cache_key = f"{DOC_VERSION}:{hashlib.sha256(doc_url.encode('utf-8')).hexdigest()}"
        
        metadata_str = await redis_client.get(doc_cache_key)
        if not metadata_str:
            raise ValueError("Document data not found in cache.")
        
        metadata = json.loads(metadata_str)
        chunks = metadata["chunks"]
        faiss_index_path = metadata["faiss_path"]
        
        faiss_index = faiss.read_index(faiss_index_path)

        expanded_queries = await semantic_expand_query(question, request_id, user_id)
        
        async with embedding_semaphore:
            embedding_model = get_embedding_model()
            query_embeddings = await asyncio.to_thread(embedding_model.encode, expanded_queries)
        
        embedding_tokens_counter.inc(sum(len(q.split()) for q in expanded_queries))

        query_embeddings = np.array(query_embeddings).astype('float32')
        faiss.normalize_L2(query_embeddings)
        
        distances, initial_indices = faiss_index.search(query_embeddings, TOP_K_CHUNKS)
        
        unique_indices = np.unique(initial_indices.flatten())
        retrieved_chunks_initial = [chunks[i] for i in unique_indices]

        if not retrieved_chunks_initial:
            return AnswerDetail(
                question=question,
                answer="I could not find a relevant answer in the document.",
                confidence_score=0.0,
                rationale="No relevant information was retrieved from the document.",
                provenance_chunks=[]
            )

        async with extractive_qa_semaphore:
            extractive_qa_pipeline = get_extractive_qa_pipeline()
            chunk_qa_tasks = [
                asyncio.to_thread(extractive_qa_pipeline, question=question, context=chunk, top_k=1)
                for chunk in retrieved_chunks_initial
            ]
            chunk_qa_results = await asyncio.gather(*chunk_qa_tasks)
        
        best_answers = [
            result[0] for result in chunk_qa_results if result[0]['score'] > 0.1
        ]
        
        if not best_answers:
            return AnswerDetail(
                question=question,
                answer="I could not find a relevant answer in the document.",
                confidence_score=0.0,
                rationale="The model could not extract a confident answer from any of the chunks.",
                provenance_chunks=retrieved_chunks_initial
            )

        best_answers.sort(key=lambda x: x['score'], reverse=True)
        combined_extractive_answers = " ".join([ans['answer'] for ans in best_answers])
        
        async with summarization_semaphore:
            summarization_pipeline = get_summarization_pipeline()
            summary = await asyncio.to_thread(summarization_pipeline, combined_extractive_answers, max_length=100, min_length=30, do_sample=False)
            summarized_context = summary[0]['summary_text']
        
        async with generative_semaphore:
            generative_model, generative_tokenizer = get_generative_llm()
            prompt_template = f"Based on the following facts, answer the question concisely:\n\nFacts: {summarized_context}\n\nQuestion: {question}\n\nAnswer: "
            inputs = generative_tokenizer(prompt_template, return_tensors="pt", max_length=MAX_PROMPT_TOKENS, truncation=True)
            generative_output = await asyncio.to_thread(
                generative_model.generate,
                inputs['input_ids'],
                max_length=150,
                num_return_sequences=1,
                early_stopping=True
            )
            final_answer = generative_tokenizer.decode(generative_output[0], skip_special_tokens=True)
            
            # Count output tokens
            generative_tokens_counter.inc(len(generative_output[0]))
        
        qa_latency_seconds.observe(time.time() - start_time)
        questions_answered_counter.inc()

        return AnswerDetail(
            question=question,
            answer=final_answer,
            confidence_score=float(best_answers[0]['score']),
            rationale=summarized_context,
            provenance_chunks=retrieved_chunks_initial
        )

    except Exception as e:
        logger.error(f"Error processing question '{question}' for {doc_url}: {e}", extra={"request_id": request_id, "user_id": user_id})
        return AnswerDetail(
            question=question,
            answer="An error occurred while processing this question.",
            confidence_score=0.0,
            rationale=str(e),
            provenance_chunks=[]
        )

# --- 7. FASTAPI ENDPOINTS ---
@app.post("/qa", response_model=AnswerResponse, status_code=200)
async def run_document_qa(request: AnswerRequest, background_tasks: BackgroundTasks, auth: dict = Depends(verify_token), req: Request = None):
    """Main endpoint to perform document-based question answering."""
    req.state.user_id = auth["user_id"]
    audit_logger.info("Authentication successful", extra={"request_id": req.state.request_id, "user_id": req.state.user_id})

    if "read" not in auth["scopes"]:
        audit_logger.warning("Authorization failed", extra={"request_id": req.state.request_id, "user_id": req.state.user_id})
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to perform this action.")
        
    doc_url = str(request.document_url)
    doc_cache_key = f"{DOC_VERSION}:{hashlib.sha256(doc_url.encode('utf-8')).hexdigest()}"
    
    try:
        if not await app.state.redis_client.exists(doc_cache_key):
            current_status = processing_status.get(doc_url, "not_found")
            if current_status == "processing":
                raise HTTPException(
                    status_code=status.HTTP_202_ACCEPTED,
                    detail="Document is already being processed. Please wait and try again."
                )
            elif current_status == "failed":
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Previous attempt to process this document failed. Please try again later."
                )
            
            background_tasks.add_task(process_document_and_build_index, doc_url, req.state.request_id, req.state.user_id)
            return AnswerResponse(
                answers=[], 
                message="Document is being processed in the background. Please try again in a moment."
            )
    except ConnectionError as e:
        logger.error(f"Redis connection failed during document check: {e}", extra={"request_id": req.state.request_id, "user_id": req.state.user_id})
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Unable to connect to document cache service.")

    tasks = [process_question_in_background(req.state.request_id, req.state.user_id, doc_url, q.question) for q in request.questions]
    
    try:
        answers = await asyncio.gather(*tasks)
        return AnswerResponse(
            answers=answers,
            message=f"Successfully answered {len(answers)} questions for document at {doc_url}."
        )
    except Exception as e:
        logger.error(f"Failed to process questions: {e}", extra={"request_id": req.state.request_id, "user_id": req.state.user_id})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while processing the questions.")
