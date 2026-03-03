"""
news_service.py
Real-time news fetching via RapidAPI Google News (google-news13.p.rapidapi.com).

Set RAPIDAPI_KEY and RAPIDAPI_NEWS_HOST in .env to enable.
If the key is absent or a placeholder, the service returns [] and the
frontend shows an empty state.
"""

import aiohttp
from datetime import datetime, timezone
from app.config import settings

# ---------------------------------------------------------------------------
# Category → keyword mapping
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS: dict[str, str] = {
    "all":             "india news today education technology science economy",
    "current_affairs": "current affairs world news",
    "science_tech":    "science technology innovation",
    "exams":           "board exam competitive exam",
    "education":       "education school university learning",
    "economy":         "economy business finance",
    "career":          "career jobs employment skills",
}

# ---------------------------------------------------------------------------
# Language → lr locale code (Google News locale param)
# ---------------------------------------------------------------------------
LANGUAGE_LOCALE: dict[str, str] = {
    "en": "en-US",
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
}

# Appending the language name to the search keyword improves
# non-English result coverage from the Google News search index
LANGUAGE_KEYWORD_HINT: dict[str, str] = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    # "en" intentionally omitted — English is the default index language
}

BASE_URL = "https://google-news13.p.rapidapi.com/search"
_PLACEHOLDER = "your_rapidapi_key_here"


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class NewsService:
    """Fetches and normalises real-time Google News articles via RapidAPI."""

    # ---- public methods ---------------------------------------------------

    async def get_common_news(
        self,
        category: str = "all",
        language: str = "en",
        max_results: int = 10,
    ) -> list[dict]:
        """Return real-time news for the given UI category."""
        if not self._key_configured():
            return []

        keyword = CATEGORY_KEYWORDS.get(category, CATEGORY_KEYWORDS["all"])
        lr      = LANGUAGE_LOCALE.get(language, "en-US")

        # Append language hint for better non-English coverage
        hint = LANGUAGE_KEYWORD_HINT.get(language)
        if hint:
            keyword = f"{keyword} {hint}"

        articles = await self._fetch(keyword=keyword, lr=lr)
        return self._normalize(articles, fallback_category=category)[:max_results]

    async def get_personal_news(
        self,
        user_context: dict,
        language: str = "en",
        max_results: int = 10,
    ) -> list[dict]:
        """
        Build a personalised search keyword from the user's learning context
        and return matching real-time news.

        user_context keys (all optional):
          subjects       list[str]  – subjects the user studies
          topics         list[str]  – actively mastered topics
          weak_topics    list[str]  – low-mastery topics (< 50%)
          grade          int | None – current grade
          board          str | None – curriculum board (CBSE, ICSE, …)
          recent_queries list[str]  – recent AI assistant queries
        """
        if not self._key_configured():
            return []

        parts: list[str] = []
        if user_context.get("subjects"):
            parts.extend(str(s) for s in user_context["subjects"][:2])
        if user_context.get("grade"):
            parts.append(f"class {user_context['grade']}")   # "class 4" not just "4"
        if user_context.get("board"):
            parts.append(str(user_context["board"]))          # e.g. "CBSE", "ICSE"
        if user_context.get("topics"):
            parts.extend(str(t) for t in user_context["topics"][:2])
        if user_context.get("weak_topics"):
            parts.extend(str(t) for t in user_context["weak_topics"][:2])
        if user_context.get("recent_queries"):
            parts.extend(str(q)[:40] for q in user_context["recent_queries"][:2])

        if not parts:
            parts = ["education", "learning"]

        keyword = " ".join(parts)[:150]
        lr      = LANGUAGE_LOCALE.get(language, "en-US")

        # Append language hint for better non-English coverage
        hint = LANGUAGE_KEYWORD_HINT.get(language)
        if hint:
            keyword = f"{keyword} {hint}"

        articles = await self._fetch(keyword=keyword, lr=lr)
        return self._normalize(articles, fallback_category="education", personal=True)[:max_results]

    # ---- private helpers --------------------------------------------------

    def _key_configured(self) -> bool:
        key = settings.RAPIDAPI_KEY
        return bool(key) and key != _PLACEHOLDER

    async def _fetch(self, keyword: str, lr: str) -> list[dict]:
        """
        Call google-news13.p.rapidapi.com/search and return the raw item list.
        """
        headers = {
            "X-RapidAPI-Key":  settings.RAPIDAPI_KEY,
            "X-RapidAPI-Host": settings.RAPIDAPI_NEWS_HOST,
        }
        params = {
            "keyword": keyword,
            "lr":      lr,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    BASE_URL,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
                    # google-news13 returns {"items": [...]}
                    return data.get("items", [])
        except Exception:
            return []

    def _normalize(
        self,
        items: list[dict],
        fallback_category: str,
        personal: bool = False,
    ) -> list[dict]:
        """
        Convert google-news13 item shape → InsightFeedCard-compatible shape.

        google-news13 item fields (approximate):
          title, snippet, publisher, timestamp (Unix ms),
          newsUrl, images: {thumbnail, thumbnailProxied}
        """
        result = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in items:
            raw_url = item.get("newsUrl") or item.get("url") or ""
            article_id = str(abs(hash(raw_url)) % (10 ** 10))

            # Prefer snippet, fall back to description or empty
            summary = (
                item.get("snippet")
                or item.get("description")
                or ""
            )[:300]

            # Prefer proxied thumbnail (avoids browser CORS issues)
            images = item.get("images") or {}
            image_url = images.get("thumbnailProxied") or images.get("thumbnail")

            # Normalise timestamp → ISO 8601
            # google-news13 returns timestamp in milliseconds (e.g. 1727222400000)
            raw_ts = item.get("timestamp")
            if raw_ts is not None:
                try:
                    ts_float = float(raw_ts)
                    # Detect milliseconds vs seconds by magnitude (> 1e10 = ms)
                    if ts_float > 1e10:
                        ts_float /= 1000
                    created_at = datetime.fromtimestamp(ts_float, tz=timezone.utc).isoformat()
                except (ValueError, TypeError, OSError):
                    created_at = str(raw_ts)
            else:
                created_at = now_iso

            result.append({
                "id":          article_id,
                "title":       item.get("title", ""),
                "summary":     summary,
                "category":    "education" if personal else fallback_category,
                "bookmarked":  False,
                "source_url":  raw_url,
                "source_name": item.get("publisher", ""),
                "image_url":   image_url,
                "created_at":  created_at,
            })
        return result
