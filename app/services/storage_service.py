import os
import uuid
import aiofiles
from pathlib import Path
from fastapi import UploadFile, HTTPException, status

from app.config import settings


class StorageService:
    """
    Local filesystem storage service.
    In production, replace with S3/GCS/Azure Blob Storage by swapping the
    upload_file / delete_file implementations.
    """

    BUCKET_DIRS = {
        "user-library": "user-library",
        "assignment-files": "assignment-files",
        "generated-ebooks": "generated-ebooks",
        "audio-responses": "audio-responses",
        "past-papers": "past-papers",
        "chat-attachments": "chat-attachments",
    }

    def _get_bucket_path(self, bucket: str) -> Path:
        dir_name = self.BUCKET_DIRS.get(bucket, bucket)
        return Path(settings.STORAGE_ROOT) / dir_name

    async def upload_file(
        self,
        file: UploadFile,
        bucket: str,
        prefix: str = "",
    ) -> dict:
        """
        Save an uploaded file to local storage.
        Returns a dict with {path, url, size_mb, type}.
        """
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must have a filename",
            )

        # Read content
        content = await file.read()
        file_size_bytes = len(content)
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Enforce size limit
        if file_size_bytes > settings.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max allowed: {settings.MAX_UPLOAD_SIZE_MB} MB",
            )

        # Build storage path
        ext = Path(file.filename).suffix
        unique_name = f"{uuid.uuid4()}{ext}"
        relative_path = os.path.join(prefix, unique_name) if prefix else unique_name

        bucket_dir = self._get_bucket_path(bucket)
        full_dir = bucket_dir / (prefix or "")
        full_dir.mkdir(parents=True, exist_ok=True)
        full_path = full_dir / unique_name

        # Write file
        async with aiofiles.open(full_path, "wb") as f:
            await f.write(content)

        # Reset file pointer for potential re-use
        await file.seek(0)

        # Build accessible URL (works for local dev with StaticFiles mounted)
        url = f"/uploads/{self.BUCKET_DIRS.get(bucket, bucket)}/{relative_path}"

        return {
            "path": str(full_path),
            "url": url,
            "size_mb": round(file_size_mb, 4),
            "type": file.content_type,
            "original_name": file.filename,
        }

    async def delete_file(self, file_path: str) -> bool:
        """Delete a file by its absolute path."""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                return True
        except Exception:
            pass
        return False

    def read_file(self, file_path: str) -> bytes | None:
        """Read file content synchronously."""
        try:
            path = Path(file_path)
            if path.exists():
                return path.read_bytes()
        except Exception:
            pass
        return None

    async def read_file_async(self, file_path: str) -> bytes | None:
        """Read file content asynchronously."""
        try:
            async with aiofiles.open(file_path, "rb") as f:
                return await f.read()
        except Exception:
            return None
