"""
news_service.py
Real-time news fetching via RapidAPI Google News (google-news13.p.rapidapi.com).

Set RAPIDAPI_KEY and RAPIDAPI_NEWS_HOST in .env to enable.
If the key is absent or a placeholder, the service returns [] and the
frontend shows an empty state.

All articles pass through a STUDENT-SAFE FILTER (see _is_student_safe) before
being returned to the frontend. Eduverse is a student-facing platform so we
hard-block headlines that mention violence, sexual content, substance abuse,
self-harm, gambling, hate, terror, scandal, or other age-inappropriate topics.
False positives are acceptable; false negatives are not.
"""

import logging
import re
import aiohttp
from datetime import datetime, timezone, timedelta
from app.config import settings

logger = logging.getLogger("news_service")

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

SEARCH_URL = "https://google-news13.p.rapidapi.com/search"
LATEST_URL = "https://google-news13.p.rapidapi.com/latest"
_PLACEHOLDER = "your_rapidapi_key_here"


# ---------------------------------------------------------------------------
# Student-safe blocklist
# ---------------------------------------------------------------------------
# Headline-shaped regex patterns. These match against the article's combined
# title + summary (lowercased). Each entry is a single-word or short-phrase
# unambiguous trigger — we accept some false positives in exchange for
# guaranteeing student-appropriate content. The grouping below is purely
# documentary; at runtime they're all flattened into _STUDENT_BLOCK_PATTERNS.
#
# DO NOT add ambiguous single words like "shooting" (matches "basketball
# shooting") or "drug" (matches "drug discovery"). Prefer specific phrases.
# ---------------------------------------------------------------------------

_STUDENT_BLOCK_RAW: list[str] = [
    # ── Sexual / explicit ────────────────────────────────────────────────
    r"\bporn(?:ography|ographic)?\b",
    r"\bsex(?:ual)?\s+(?:assault|abuse|harassment|misconduct|offender|predator|crime|violence|slavery|trafficking)\b",
    r"\brape(?:d|ist|s)?\b",
    r"\bmolest(?:ed|ing|ation)?\b",
    r"\bpaedophile|pedophile|paedophilia|pedophilia\b",
    r"\bchild\s+(?:abuse|exploitation|marriage|bride|porn|sex)\b",
    r"\bcsam\b",
    r"\bobscene|obscenity|nudity|nude\s+(?:photos?|video|leak)\b",
    r"\bonlyfans\b",
    r"\bescort\s+(?:service|agency|girl)\b",
    r"\bsex\s+(?:tape|scandal|racket|worker|toy)\b",
    r"\bincest\b",
    r"\baphrodisiac\b",
    r"\berotic(?:a)?\b",

    # ── Graphic violence / crime ─────────────────────────────────────────
    r"\bmurder(?:ed|er|ous)?\b",
    r"\bhomicide\b",
    r"\bmassacre(?:d|s)?\b",
    r"\blynch(?:ed|ing|ings)?\b",
    r"\bstabb(?:ed|ing|s)\b",
    r"\bshot\s+dead\b",
    r"\bgunned\s+down\b",
    r"\bmass\s+shooting\b",
    r"\bschool\s+shooting\b",
    r"\bgang\s*(?:rape|war|fight|murder)\b",
    r"\bdecapitat(?:ed|ion)\b",
    r"\bbeheading|beheaded\b",
    r"\btorture(?:d|s)?\b",
    r"\bdismember(?:ed|ing|ment)?\b",
    r"\bbeat(?:en)?\s+to\s+death\b",
    r"\bbody\s+(?:parts?|found|recovered|dumped)\b",
    r"\bcorpse|cadaver\b",
    r"\bbloodbath|bloodshed\b",
    r"\bassassinat(?:ed|ion)\b",
    r"\bhostage(?:s)?\b",
    r"\bkidnap(?:ped|ping|s)?\b",
    r"\bhuman\s+trafficking\b",
    r"\bhonou?r\s+killing\b",
    r"\bdowry\s+death\b",
    r"\bacid\s+attack\b",

    # ── Self-harm / suicide ──────────────────────────────────────────────
    r"\bsuicide(?:d|s)?\b",
    r"\bself[-\s]?harm(?:ing)?\b",
    r"\bhang(?:ed|ing)\s+(?:self|himself|herself)\b",
    r"\bjumps?\s+(?:from|off)\s+(?:building|bridge|terrace)\b",
    r"\bslit(?:s|ting)?\s+(?:wrist|throat)\b",

    # ── Substance abuse ──────────────────────────────────────────────────
    r"\bdrug\s+(?:bust|raid|cartel|trafficking|peddler|smuggling|dealer|overdose|abuse|addict)\b",
    r"\bnarcotic(?:s)?\b",
    r"\bcocaine|heroin|methamphetamine|\bmeth\b|fentanyl|opioid|opium|ketamine|lsd|ecstasy|mdma|marijuana|cannabis|weed\s+seized|hashish|ganja\b",
    r"\bvaping\s+(?:death|illness|injury)\b",
    r"\balcohol(?:ism)?\s+(?:abuse|addict|death|poisoning)\b",
    r"\bhooch\s+tragedy\b",
    r"\bliquor\s+(?:tragedy|death)\b",

    # ── Terror / war atrocities ──────────────────────────────────────────
    r"\bterror(?:ist|ism|ists)\b",
    r"\bsuicide\s+bomb(?:er|ing|ings)?\b",
    r"\bbomb\s+(?:blast|attack|explosion)\b",
    r"\bgrenade\s+attack\b",
    r"\bairstrike\s+(?:kills?|deaths?)\b",
    r"\bdrone\s+strike\s+(?:kills?|deaths?)\b",
    r"\bgenocide\b",
    r"\bethnic\s+cleansing\b",
    r"\bwar\s+crime(?:s)?\b",
    r"\bcivilian\s+(?:casualt|deaths?|killed)\b",
    r"\bisis\b|\bisil\b|\bal[-\s]?qaeda\b|\btaliban\s+attack\b",

    # ── Hate / communal ──────────────────────────────────────────────────
    r"\bhate\s+(?:crime|speech)\b",
    r"\bcommunal\s+(?:violence|riot|clash|tension)\b",
    r"\bracist\s+(?:attack|slur)\b",
    r"\bhomophobic\s+attack\b",
    r"\blgbt(?:q)?\+?\s+(?:slur|attack)\b",

    # ── Gambling / betting ───────────────────────────────────────────────
    r"\bbetting\s+(?:racket|app|scandal|scam)\b",
    r"\bgambl(?:ing|er)\s+(?:den|racket|addict)\b",
    r"\bcasino\s+(?:raid|scandal)\b",
    r"\bmatch[-\s]?fixing\b",
    r"\bipl\s+betting\b",

    # ── Sensational / scandal / gossip — non-educational ────────────────
    r"\bscandal\b",
    r"\bcontroversy\b",
    r"\bcontroversial\s+(?:remark|statement|comment|tweet|post)\b",
    r"\bdivorce\s+(?:drama|scandal|battle)\b",
    r"\bdating\s+(?:rumou?rs?|drama)\b",
    r"\bcheating\s+(?:scandal|allegation)\b",
    r"\bmms\s+(?:leak|scandal)\b",
    r"\bleaked\s+(?:photos?|video|chat|nude|private)\b",
    r"\bviral\s+(?:fight|brawl|slap|kiss|nude)\b",

    # ── Misc adult / inappropriate ───────────────────────────────────────
    r"\bbrothel\b",
    r"\bstrip\s+club\b",
    r"\bsex\s+toy\b",
    r"\bcondom\s+(?:ad|brand|scandal)\b",
    r"\bdating\s+app\s+(?:scandal|murder|crime)\b",
]

_STUDENT_BLOCK_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _STUDENT_BLOCK_RAW
]


def _is_student_safe(item: dict) -> tuple[bool, str | None]:
    """Return (allowed, trigger). False ⇒ drop the article from the feed.

    Scans the title and summary together. We deliberately keep this synchronous
    and regex-only so it costs ~zero per call and can run on every fetched
    article without making the news endpoint slow.
    """
    haystack_parts = [
        str(item.get("title") or ""),
        str(item.get("snippet") or item.get("description") or ""),
    ]
    haystack = " ".join(haystack_parts).lower()
    if not haystack.strip():
        # Headlines without text are useless; drop them too.
        return False, "empty_text"

    for pat in _STUDENT_BLOCK_PATTERNS:
        m = pat.search(haystack)
        if m:
            return False, m.group(0)[:50]
    return True, None


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

def _recency_hint() -> str:
    """Return a recency hint covering the last 30 days (e.g. 'February March 2026').

    On day 1 of a month, using only the current month name would miss the
    previous month's news, so we always include both the current and the
    month-from-30-days-ago names.
    """
    now = datetime.now()
    ago = now - timedelta(days=30)
    months = sorted(set([ago.strftime("%B"), now.strftime("%B")]))
    return f"{' '.join(months)} {now.year}"


class NewsService:
    """Fetches and normalises real-time Google News articles via RapidAPI."""

    # ---- public methods ---------------------------------------------------

    async def get_common_news(
        self,
        category: str = "all",
        language: str = "en",
        max_results: int = 10,
    ) -> list[dict]:
        """Return real-time news for the given UI category.

        Uses /latest for today's headlines (``all`` category) and /search with
        current-year hint for category-specific results.
        Results are always sorted newest-first.
        """
        if not self._key_configured():
            return []

        lr = LANGUAGE_LOCALE.get(language, "en-US")

        if category == "all":
            # Use /latest for today's top headlines
            articles = await self._fetch_latest(lr=lr)
            return self._normalize(articles, fallback_category="all")[:max_results]

        # For specific categories — append recency hint for freshness
        keyword = CATEGORY_KEYWORDS.get(category, CATEGORY_KEYWORDS["all"])
        keyword = f"{keyword} {_recency_hint()}"
        hint = LANGUAGE_KEYWORD_HINT.get(language)
        if hint:
            keyword = f"{keyword} {hint}"

        articles = await self._fetch_search(keyword=keyword, lr=lr)
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
            logger.warning("[news] RAPIDAPI_KEY not configured — personal news returning []")
            return []

        lr = LANGUAGE_LOCALE.get(language, "en-US")
        hint = LANGUAGE_KEYWORD_HINT.get(language)
        recency = _recency_hint()

        # A single compound keyword jamming every context signal together
        # (subject + grade + board + topics + weak topics + recent queries)
        # reads nothing like a real headline, so Google News search — a literal
        # phrase match, not a weighted multi-field OR — frequently returns zero
        # results for real (non-empty-context) users even though the API/key
        # is perfectly healthy. Try progressively broader candidates instead of
        # gambling everything on one shot, so users reliably see *something*.
        for keyword in self._build_keyword_candidates(user_context):
            full_keyword = f"{keyword} {recency}"
            if hint:
                full_keyword = f"{full_keyword} {hint}"

            ok, raw_items = await self._raw_fetch_with_status(
                SEARCH_URL, {"keyword": full_keyword[:150], "lr": lr}
            )
            if not ok:
                # Hard failure (bad key, quota exhausted, network/timeout) — retrying
                # with a different keyword against the same broken call won't help.
                logger.warning("[news] personal search hard-failed, aborting cascade (keyword=%r)", full_keyword)
                return []

            normalized = self._normalize(raw_items, fallback_category="education", personal=True)
            if normalized:
                return normalized[:max_results]
            logger.info("[news] personal keyword matched 0 articles, trying broader candidate (keyword=%r)", full_keyword)

        # Every candidate — down to the generic "education learning" fallback —
        # matched nothing. Fall back to today's general top headlines rather
        # than ever showing the user a blank feed.
        ok, raw_items = await self._raw_fetch_with_status(LATEST_URL, {"lr": lr})
        if not ok:
            return []
        return self._normalize(raw_items, fallback_category="education", personal=True)[:max_results]

    def _build_keyword_candidates(self, user_context: dict) -> list[str]:
        """Ordered search-keyword candidates, most specific first.

        Each is short enough to plausibly match a real headline — unlike the
        old single compound-everything keyword, which almost never did.
        """
        subjects = [str(s) for s in (user_context.get("subjects") or [])]
        grade = user_context.get("grade")
        board = user_context.get("board")
        topics = [str(t) for t in (user_context.get("topics") or [])]
        weak_topics = [str(t) for t in (user_context.get("weak_topics") or [])]
        recent_queries = [str(q)[:60] for q in (user_context.get("recent_queries") or [])]

        candidates: list[str] = []

        def _add(parts: list[str]):
            joined = " ".join(p for p in parts if p).strip()
            if joined and joined not in candidates:
                candidates.append(joined)

        # 1. Most specific: subject + grade + board + top weak/active topic
        _add([
            subjects[0] if subjects else "",
            f"class {grade}" if grade else "",
            str(board) if board else "",
            (weak_topics or topics)[0] if (weak_topics or topics) else "",
        ])
        # 2. Subject + grade/board only
        _add([
            subjects[0] if subjects else "",
            f"class {grade}" if grade else "",
            str(board) if board else "",
        ])
        # 3. Just the subject
        if subjects:
            _add([subjects[0]])
        # 4. A recent AI-assistant query on its own — often more newsworthy
        #    standalone than buried inside a compound string.
        if recent_queries:
            _add([recent_queries[0]])
        # 5. Generic education fallback — always tried last, guaranteed non-empty
        _add(["education", "learning"])

        return candidates

    # ---- private helpers --------------------------------------------------

    def _key_configured(self) -> bool:
        key = settings.RAPIDAPI_KEY
        return bool(key) and key != _PLACEHOLDER

    async def _fetch_search(self, keyword: str, lr: str) -> list[dict]:
        """Call /search endpoint — keyword-based results (may not be the freshest)."""
        _, items = await self._raw_fetch_with_status(SEARCH_URL, {"keyword": keyword, "lr": lr})
        return items

    async def _fetch_latest(self, lr: str) -> list[dict]:
        """Call /latest endpoint — today's top headlines (always fresh)."""
        _, items = await self._raw_fetch_with_status(LATEST_URL, {"lr": lr})
        return items

    async def _raw_fetch(self, url: str, params: dict) -> list[dict]:
        """Generic HTTP GET against the RapidAPI host. Kept for callers that don't
        need to distinguish a hard failure from a genuine zero-match result."""
        _, items = await self._raw_fetch_with_status(url, params)
        return items

    async def _raw_fetch_with_status(self, url: str, params: dict) -> tuple[bool, list[dict]]:
        """Generic HTTP GET against the RapidAPI host.

        Returns (ok, items). ok=False means a hard failure (bad key, quota
        exhausted, network/timeout error) — as opposed to a successful call
        that simply matched zero articles — logged with enough detail to
        diagnose from server logs without needing another DevTools round trip.
        """
        headers = {
            "X-RapidAPI-Key":  settings.RAPIDAPI_KEY,
            "X-RapidAPI-Host": settings.RAPIDAPI_NEWS_HOST,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning(
                            "[news] RapidAPI request failed: status=%s url=%s params=%s body=%s",
                            resp.status, url, params, body[:300],
                        )
                        return False, []
                    data = await resp.json()
                    return True, data.get("items", [])
        except Exception as exc:
            logger.warning("[news] RapidAPI request errored: url=%s params=%s error=%r", url, params, exc)
            return False, []

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
        dropped = 0

        for item in items:
            # Student-safety filter — drop anything that mentions violence,
            # explicit content, substance abuse, terror, gambling, scandal, etc.
            # Eduverse is a student platform; we err on the side of caution.
            allowed, trigger = _is_student_safe(item)
            if not allowed:
                dropped += 1
                logger.info(
                    "[news] dropped unsafe article (trigger=%s): %s",
                    trigger, (item.get("title") or "")[:120],
                )
                continue

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

        # Sort by date — newest first
        result.sort(key=lambda a: a["created_at"], reverse=True)
        if dropped:
            logger.info(
                "[news] student-safe filter dropped %d/%d articles (kept %d)",
                dropped, len(items), len(result),
            )
        return result
