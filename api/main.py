"""
FastAPI Application for DocuMind

Provides REST API endpoints for document processing and Q&A.
"""

import os
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from documind import DocuMind


# Pydantic Models
class QueryRequest(BaseModel):
    """Request model for document queries."""
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: Optional[List[str]] = None


class QueryResponse(BaseModel):
    """Response model for document queries."""
    answer: str
    sources: List[dict]
    confidence: float
    processing_time: float


class DocumentResponse(BaseModel):
    """Response model for document upload."""
    document_id: str
    filename: str
    num_chunks: int
    status: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str


# Initialize FastAPI app
app = FastAPI(
    title="DocuMind API",
    description="Multimodal RAG Document Q&A System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DocuMind instance
documind = DocuMind()


# Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.
    
    Supports PDF, PNG, JPG, JPEG, TIFF, and BMP files.
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file
        upload_dir = os.getenv("UPLOAD_DIR", "./data/uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process document
        result = documind.load_document(file_path)
        
        return DocumentResponse(
            document_id=result["document_id"],
            filename=file.filename,
            num_chunks=result["num_chunks"],
            status="success",
            message=f"Document processed successfully with {result['num_chunks']} chunks"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the document collection.
    
    Returns an answer based on the indexed documents.
    """
    import time
    start_time = time.time()
    
    try:
        response = documind.query(
            question=request.question,
            top_k=request.top_k
        )
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/documents")
async def list_documents():
    """List all indexed documents."""
    return {
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "num_chunks": doc.num_chunks,
                "created_at": doc.created_at.isoformat()
            }
            for doc in documind.documents
        ],
        "total": len(documind.documents)
    }


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a specific document."""
    # TODO: Implement document deletion
    raise HTTPException(
        status_code=501,
        detail="Document deletion not yet implemented"
    )


@app.post("/clear")
async def clear_all():
    """Clear all indexed documents."""
    documind.clear()
    return {"status": "success", "message": "All documents cleared"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )
