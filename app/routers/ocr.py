import re
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from app.dependencies import DBSession, CurrentUser
from app.models.content import UserLibraryItem, DocChunk
from app.services.ai_service import AIService
from app.services.storage_service import StorageService
from app.services.points_service import PointsService
from app.schemas.content import OCRExtractResponse

router = APIRouter()

# Regex to strip null bytes and non-printable control chars (keep \n \r \t)
_INVALID_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _parse_org_id(org_id: str | None) -> uuid.UUID | None:
    if not org_id or org_id == "personal":
        return None
    try:
        return uuid.UUID(org_id)
    except ValueError:
        return None


@router.post("/extract", response_model=OCRExtractResponse)
async def extract_text(
    file: UploadFile = File(...),
    language: str = Form("en"),
    org_id: str = Form(None),
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
            detail="This file type is not supported for OCR. Please upload a PDF, JPEG, PNG, GIF, or WebP file.",
        )

    # Check usage limits and deduct points (cost=3, xp=3 from seed)
    parsed_org = _parse_org_id(org_id)
    points_service = PointsService()
    await points_service.check_and_increment_usage(
        user_id=current_user.id, feature_key="ocr_extraction", db=db, org_id=parsed_org,
    )
    await points_service.deduct(
        user_id=current_user.id, action="ocr_extraction", db=db, org_id=parsed_org,
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
        processing_status="ready",
        org_id=_parse_org_id(org_id),
    )
    db.add(item)
    await db.flush()

    # Sanitize: strip null bytes and control chars that PostgreSQL rejects
    if extracted_text:
        extracted_text = _INVALID_CHARS.sub('', extracted_text)

    # Store extracted text as chunks for later retrieval via GET /library/{id}/text
    if extracted_text:
        chunks = ai.chunk_text(extracted_text)
        for i, chunk in enumerate(chunks):
            db.add(DocChunk(
                library_item_id=item.id,
                chunk_text=_INVALID_CHARS.sub('', chunk),
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
