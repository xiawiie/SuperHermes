import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.security.auth import require_admin
from backend.infra.db.models import User
from backend.contracts.schemas import DocumentDeleteResponse, DocumentInfo, DocumentListResponse, DocumentUploadResponse
from backend.services.document_service import DocumentProcessingError, DocumentService

_document_service: DocumentService | None = None

router = APIRouter()


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        _document_service = DocumentService.create_default()
    return _document_service


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(_: User = Depends(require_admin)):
    """List indexed documents for administrators."""
    try:
        service = get_document_service()
        documents = [DocumentInfo(**stats) for stats in await asyncio.to_thread(service.list_documents)]
        return DocumentListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), _: User = Depends(require_admin)):
    """Upload a document and index it for retrieval."""
    try:
        filename = file.filename or ""
        file_lower = filename.lower()
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        if not (
            file_lower.endswith(".pdf")
            or file_lower.endswith((".docx", ".doc"))
            or file_lower.endswith((".xlsx", ".xls"))
        ):
            raise HTTPException(status_code=400, detail="Only PDF, Word, and Excel files are supported")

        try:
            service = get_document_service()
            content = await file.read()
            result = await asyncio.to_thread(service.upload_document, filename, content)
        except DocumentProcessingError as e:
            raise HTTPException(status_code=500, detail=str(e))
        return DocumentUploadResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str, _: User = Depends(require_admin)):
    """Delete a document from Milvus and parent chunk storage."""
    try:
        service = get_document_service()
        result = await asyncio.to_thread(service.delete_document, filename)
        return DocumentDeleteResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
