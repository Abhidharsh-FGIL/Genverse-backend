from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from app.dependencies import DBSession, CurrentUser
from app.models.content import UserLibraryItem, DocChunk
from app.services.ai_service import AIService
from app.services.storage_service import StorageService
from app.schemas.content import OCRExtractResponse

router = APIRouter()


@router.post("/extract", response_model=OCRExtractResponse)
async def extract_text(
    file: UploadFile = File(...),
    language: str = Form("en"),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """Extract text from an uploaded image or document using OCR and save it to the library."""
    allowed_types = {
        "image/jpeg", "image/png", "image/gif", "image/webp",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type for OCR",
        )

    storage = StorageService()
    file_info = await storage.upload_file(
        file=file,
        bucket="user-library",
        prefix=f"{current_user.id}/ocr",
    )

    ai = AIService()
    extracted_text = await ai.extract_text_from_file(
        file_path=file_info["path"],
        language=language,
    )

    # Save library item so it appears in GET /library?folder=ocr
    item = UserLibraryItem(
        user_id=current_user.id,
        title=file.filename or "OCR Document",
        file_type=file_info.get("type") or file.content_type,
        storage_path=file_info.get("path"),
        file_size_mb=file_info.get("size_mb"),
        folder="ocr",
        is_processed=bool(extracted_text),
    )
    db.add(item)
    await db.flush()

    # Store extracted text as chunks for later retrieval via GET /library/{id}/text
    if extracted_text:
        chunks = ai.chunk_text(extracted_text)
        for i, chunk in enumerate(chunks):
            db.add(DocChunk(
                library_item_id=item.id,
                chunk_text=chunk,
                chunk_order=i,
            ))

    await db.commit()
    await db.refresh(item)

    return OCRExtractResponse(
        item=item,
        extracted_text=extracted_text or "",
        word_count=len(extracted_text.split()) if extracted_text else 0,
        language=language,
    )
