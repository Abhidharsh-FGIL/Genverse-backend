"""YouTube Data API v3 search service."""
from typing import List
import httpx
from app.config import settings


class YouTubeService:
    BASE_URL = "https://www.googleapis.com/youtube/v3"

    async def search_videos(self, query: str, max_results: int = 3) -> List[dict]:
        """Search YouTube for videos matching the query. Returns [] if API key is not configured."""
        if not settings.YOUTUBE_API_KEY:
            return []

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.BASE_URL}/search",
                    params={
                        "part": "snippet",
                        "q": query,
                        "type": "video",
                        "maxResults": max_results,
                        "key": settings.YOUTUBE_API_KEY,
                        "relevanceLanguage": "en",
                        "safeSearch": "strict",
                        "videoEmbeddable": "true",
                    },
                )
                if resp.status_code != 200:
                    return []

                data = resp.json()
                videos = []
                for item in data.get("items", []):
                    video_id = item.get("id", {}).get("videoId")
                    if not video_id:
                        continue
                    snippet = item.get("snippet", {})
                    thumbnails = snippet.get("thumbnails", {})
                    thumbnail = (
                        thumbnails.get("medium", {}).get("url")
                        or thumbnails.get("default", {}).get("url")
                        or ""
                    )
                    videos.append({
                        "title": snippet.get("title", ""),
                        "channel": snippet.get("channelTitle", ""),
                        "thumbnail": thumbnail,
                        "video_id": video_id,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                    })
                return videos
        except Exception:
            return []
