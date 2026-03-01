"""
Industry-grade audiobook generation service.

Uses edge-tts (Microsoft Azure Neural TTS) for high-quality voice synthesis
with chapter-aware narration, structured audio segments, silence gaps,
and chapter timestamp tracking.

Falls back to gTTS when edge-tts is unavailable.
"""

import asyncio
import os
import re
import uuid as _uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

from app.config import settings


# ─── Voice profiles ──────────────────────────────────────────────────────────

# Maps language code → dict of profile_name → edge-tts voice ID
VOICE_MAP: dict[str, dict[str, str]] = {
    "en": {
        "female_warm": "en-US-AriaNeural",
        "female_professional": "en-US-JennyNeural",
        "male_warm": "en-US-GuyNeural",
        "male_professional": "en-US-DavisNeural",
        "narrator": "en-US-ChristopherNeural",
    },
    "hi": {
        "female_warm": "hi-IN-SwaraNeural",
        "male_warm": "hi-IN-MadhurNeural",
        "narrator": "hi-IN-MadhurNeural",
    },
    "ta": {
        "female_warm": "ta-IN-PallaviNeural",
        "male_warm": "ta-IN-ValluvarNeural",
        "narrator": "ta-IN-ValluvarNeural",
    },
    "te": {
        "female_warm": "te-IN-ShrutiNeural",
        "male_warm": "te-IN-MohanNeural",
        "narrator": "te-IN-MohanNeural",
    },
    "es": {
        "female_warm": "es-ES-ElviraNeural",
        "male_warm": "es-ES-AlvaroNeural",
        "narrator": "es-ES-AlvaroNeural",
    },
    "fr": {
        "female_warm": "fr-FR-DeniseNeural",
        "male_warm": "fr-FR-HenriNeural",
        "narrator": "fr-FR-HenriNeural",
    },
    "de": {
        "female_warm": "de-DE-KatjaNeural",
        "male_warm": "de-DE-ConradNeural",
        "narrator": "de-DE-ConradNeural",
    },
    "ar": {
        "female_warm": "ar-SA-ZariyahNeural",
        "male_warm": "ar-SA-HamedNeural",
        "narrator": "ar-SA-HamedNeural",
    },
    "zh": {
        "female_warm": "zh-CN-XiaoxiaoNeural",
        "male_warm": "zh-CN-YunxiNeural",
        "narrator": "zh-CN-YunxiNeural",
    },
    "ja": {
        "female_warm": "ja-JP-NanamiNeural",
        "male_warm": "ja-JP-KeitaNeural",
        "narrator": "ja-JP-KeitaNeural",
    },
    "ko": {
        "female_warm": "ko-KR-SunHiNeural",
        "male_warm": "ko-KR-InJoonNeural",
        "narrator": "ko-KR-InJoonNeural",
    },
    "pt": {
        "female_warm": "pt-BR-FranciscaNeural",
        "male_warm": "pt-BR-AntonioNeural",
        "narrator": "pt-BR-AntonioNeural",
    },
}

# Narration style presets for SSML prosody
NARRATION_STYLES: dict[str, dict[str, str]] = {
    "standard": {"rate": "+0%", "pitch": "+0Hz"},
    "slow_clear": {"rate": "-15%", "pitch": "+0Hz"},
    "energetic": {"rate": "+10%", "pitch": "+2Hz"},
    "calm": {"rate": "-10%", "pitch": "-2Hz"},
}

# Words per minute for duration estimation by narration style
WPM_BY_STYLE: dict[str, int] = {
    "standard": 150,
    "slow_clear": 125,
    "energetic": 165,
    "calm": 130,
}


def get_available_voices(language: str) -> list[dict[str, str]]:
    """Return available voice profiles for a language."""
    voices = VOICE_MAP.get(language, VOICE_MAP.get("en", {}))
    return [
        {"id": profile_id, "name": profile_id.replace("_", " ").title(), "voice_id": voice_id}
        for profile_id, voice_id in voices.items()
    ]


def _resolve_voice(language: str, voice_profile: str | None) -> str:
    """Resolve a voice profile name to an edge-tts voice ID."""
    lang_voices = VOICE_MAP.get(language, VOICE_MAP.get("en", {}))
    if voice_profile and voice_profile in lang_voices:
        return lang_voices[voice_profile]
    # Default to narrator, then first available
    return lang_voices.get("narrator", next(iter(lang_voices.values()), "en-US-AriaNeural"))


# ─── Text preprocessing ─────────────────────────────────────────────────────

def clean_narration_text(text: str) -> str:
    """Clean text for natural-sounding narration."""
    if not text:
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Strip markdown formatting
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)      # bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)            # italic
    text = re.sub(r"__(.+?)__", r"\1", text)             # bold alt
    text = re.sub(r"_(.+?)_", r"\1", text)               # italic alt
    text = re.sub(r"`(.+?)`", r"\1", text)               # inline code
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # list markers
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)  # numbered list markers
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)          # images
    text = re.sub(r"\[(.+?)\]\(.*?\)", r"\1", text)      # links → keep text
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)  # blockquotes
    text = re.sub(r"---+|===+|\*\*\*+", "", text)       # horizontal rules
    text = re.sub(r"```[\s\S]*?```", "", text)           # code blocks
    text = re.sub(r"\|.*?\|", "", text)                  # table rows

    # Expand common abbreviations
    abbreviations = {
        "e.g.": "for example",
        "i.e.": "that is",
        "etc.": "etcetera",
        "vs.": "versus",
        "Dr.": "Doctor",
        "Mr.": "Mister",
        "Mrs.": "Missus",
        "Ms.": "Miss",
        "Prof.": "Professor",
        "Fig.": "Figure",
        "approx.": "approximately",
    }
    for abbr, expansion in abbreviations.items():
        text = text.replace(abbr, expansion)

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text


# ─── Narration script builder ───────────────────────────────────────────────

class NarrationSegment:
    """A single narration segment (chapter, intro, etc.) with metadata."""

    def __init__(self, label: str, text: str, segment_type: str = "chapter"):
        self.label = label
        self.text = text
        self.segment_type = segment_type  # "intro", "chapter", "summary", "outro"
        self.duration_seconds: int = 0
        self.start_seconds: int = 0


def build_narration_script(ebook_json: dict) -> list[NarrationSegment]:
    """Build a structured narration script from eBook JSON."""
    segments: list[NarrationSegment] = []

    title = ebook_json.get("title", "")
    # Try title_page first (enhanced eBook format)
    title_page = ebook_json.get("title_page", {})
    if title_page:
        title = title_page.get("title", title)
        author = title_page.get("author", "")
    else:
        author = ebook_json.get("author", "")

    # ── Intro segment
    intro_lines = [f"{title}."]
    if author:
        intro_lines.append(f"Written by {author}.")

    # Book description / about
    about = ebook_json.get("about_the_book", {})
    if isinstance(about, dict):
        desc = about.get("description", "")
    elif isinstance(about, str):
        desc = about
    else:
        desc = ""
    if desc:
        intro_lines.append(clean_narration_text(desc))

    segments.append(NarrationSegment(
        label="Introduction",
        text=" ".join(intro_lines),
        segment_type="intro",
    ))

    # ── Book summary (if present)
    book_summary = ebook_json.get("book_summary", {})
    if isinstance(book_summary, dict) and book_summary.get("content"):
        segments.append(NarrationSegment(
            label="Book Summary",
            text=clean_narration_text(book_summary["content"]),
            segment_type="summary",
        ))

    # ── Chapters
    chapters = ebook_json.get("chapters", [])
    for i, ch in enumerate(chapters, 1):
        ch_title = ch.get("title", f"Chapter {i}")
        parts: list[str] = [f"Chapter {i}. {ch_title}."]

        # Chapter content
        content = ch.get("content") or ch.get("description") or ""
        if content:
            parts.append(clean_narration_text(content))

        # Key points
        key_points = ch.get("key_points", [])
        if key_points:
            parts.append("Key points from this chapter.")
            for kp in key_points:
                if isinstance(kp, str):
                    parts.append(clean_narration_text(kp))
                elif isinstance(kp, dict):
                    parts.append(clean_narration_text(kp.get("text", kp.get("point", ""))))

        # Chapter summary
        ch_summary = ch.get("summary", "")
        if ch_summary:
            parts.append(f"In summary, {clean_narration_text(ch_summary)}")

        segments.append(NarrationSegment(
            label=f"Chapter {i}: {ch_title}",
            text=" ".join(parts),
            segment_type="chapter",
        ))

    # ── Outro
    thank_you = ebook_json.get("thank_you", {})
    if isinstance(thank_you, dict):
        outro_text = thank_you.get("message") or thank_you.get("content", "")
    elif isinstance(thank_you, str):
        outro_text = thank_you
    else:
        outro_text = ""

    if not outro_text:
        outro_text = f"This concludes {title}. Thank you for listening."
    else:
        outro_text = clean_narration_text(outro_text)

    segments.append(NarrationSegment(
        label="Closing",
        text=outro_text,
        segment_type="outro",
    ))

    return segments


# ─── Silence generation ─────────────────────────────────────────────────────

def _generate_silence_mp3(duration_ms: int) -> bytes:
    """Generate silent MP3 audio of specified duration using pydub."""
    from pydub import AudioSegment
    from pydub.generators import Sine

    # Generate silence
    silence = AudioSegment.silent(duration=duration_ms, frame_rate=24000)
    buf = BytesIO()
    silence.export(buf, format="mp3", bitrate="48k")
    return buf.getvalue()


# Silence durations (ms)
PAUSE_AFTER_INTRO = 1500
PAUSE_BETWEEN_CHAPTERS = 2000
PAUSE_AFTER_SUMMARY = 1500
PAUSE_BEFORE_OUTRO = 2000


# ─── Edge-TTS synthesis ─────────────────────────────────────────────────────

async def _synthesize_edge_tts(
    text: str,
    voice: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> bytes:
    """Synthesize text to MP3 using edge-tts (Microsoft Neural TTS)."""
    import edge_tts

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
    )

    buf = BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])

    return buf.getvalue()


async def _synthesize_gtts_fallback(text: str, lang: str) -> bytes:
    """Fallback TTS using gTTS when edge-tts fails."""
    from gtts import gTTS
    from starlette.concurrency import run_in_threadpool

    def _do():
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()

    return await run_in_threadpool(_do)


# ─── Audio concatenation ────────────────────────────────────────────────────

def _concat_audio_segments(segments_bytes: list[bytes]) -> bytes:
    """Concatenate multiple MP3 byte segments into one using pydub."""
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    for seg_bytes in segments_bytes:
        if not seg_bytes:
            continue
        seg = AudioSegment.from_mp3(BytesIO(seg_bytes))
        combined += seg

    buf = BytesIO()
    combined.export(buf, format="mp3", bitrate="128k")
    return buf.getvalue()


def _get_audio_duration_seconds(mp3_bytes: bytes) -> int:
    """Get duration of MP3 audio in seconds using pydub."""
    from pydub import AudioSegment

    audio = AudioSegment.from_mp3(BytesIO(mp3_bytes))
    return int(audio.duration_seconds)


# ─── Main generation ────────────────────────────────────────────────────────

async def generate_audiobook(
    ebook_json: dict | None,
    language: str = "en",
    voice_profile: str | None = None,
    narration_style: str = "standard",
) -> dict:
    """
    Generate an industry-grade audiobook from eBook JSON content.

    Returns:
        dict with keys:
            - audio_path: str - path to saved MP3 file
            - duration_seconds: int - total duration
            - chapter_timestamps: list[dict] - per-chapter timestamps for navigation
            - voice_used: str - the edge-tts voice ID used
            - narration_style: str - the narration style applied
    """
    if not ebook_json:
        return {"audio_path": None, "duration_seconds": None, "chapter_timestamps": []}

    # Resolve voice and style
    voice_id = _resolve_voice(language, voice_profile)
    style = NARRATION_STYLES.get(narration_style, NARRATION_STYLES["standard"])
    rate = style["rate"]
    pitch = style["pitch"]
    wpm = WPM_BY_STYLE.get(narration_style, 150)

    # Build structured narration script
    narration_segments = build_narration_script(ebook_json)

    if not narration_segments:
        return {"audio_path": None, "duration_seconds": None, "chapter_timestamps": []}

    # Synthesize each segment
    use_edge_tts = True
    audio_parts: list[bytes] = []
    chapter_timestamps: list[dict] = []
    current_offset_seconds = 0

    # gTTS language mapping
    gtts_lang_map = {
        "en": "en", "hi": "hi", "ta": "ta", "te": "te",
        "es": "es", "fr": "fr", "de": "de", "ar": "ar",
        "zh": "zh-CN", "ja": "ja", "ko": "ko", "pt": "pt",
    }
    gtts_lang = gtts_lang_map.get(language, "en")

    for i, segment in enumerate(narration_segments):
        # Skip empty segments
        if not segment.text.strip():
            continue

        # Synthesize audio for this segment
        try:
            if use_edge_tts:
                seg_audio = await _synthesize_edge_tts(
                    text=segment.text,
                    voice=voice_id,
                    rate=rate,
                    pitch=pitch,
                )
            else:
                seg_audio = await _synthesize_gtts_fallback(segment.text, gtts_lang)
        except Exception:
            # If edge-tts fails on first segment, fall back to gTTS for all
            if use_edge_tts:
                use_edge_tts = False
                try:
                    seg_audio = await _synthesize_gtts_fallback(segment.text, gtts_lang)
                except Exception:
                    continue
            else:
                continue

        if not seg_audio:
            continue

        # Record timestamp
        word_count = len(segment.text.split())
        estimated_duration = max(1, int(word_count / wpm * 60))

        chapter_timestamps.append({
            "label": segment.label,
            "type": segment.segment_type,
            "start_seconds": current_offset_seconds,
            "duration_seconds": estimated_duration,
        })

        audio_parts.append(seg_audio)
        current_offset_seconds += estimated_duration

        # Add silence between segments
        if i < len(narration_segments) - 1:
            pause_ms = PAUSE_BETWEEN_CHAPTERS
            if segment.segment_type == "intro":
                pause_ms = PAUSE_AFTER_INTRO
            elif segment.segment_type == "summary":
                pause_ms = PAUSE_AFTER_SUMMARY

            next_seg = narration_segments[i + 1] if i + 1 < len(narration_segments) else None
            if next_seg and next_seg.segment_type == "outro":
                pause_ms = PAUSE_BEFORE_OUTRO

            try:
                silence = _generate_silence_mp3(pause_ms)
                audio_parts.append(silence)
                current_offset_seconds += pause_ms / 1000
            except Exception:
                pass  # Skip silence if pydub isn't available

    if not audio_parts:
        return {"audio_path": None, "duration_seconds": None, "chapter_timestamps": []}

    # Concatenate all segments
    from starlette.concurrency import run_in_threadpool

    try:
        final_audio = await run_in_threadpool(_concat_audio_segments, audio_parts)
    except Exception:
        # If pydub concat fails, just use the raw segments concatenated
        final_audio = b"".join(audio_parts)

    # Calculate actual duration
    try:
        total_duration = await run_in_threadpool(_get_audio_duration_seconds, final_audio)
    except Exception:
        total_duration = int(current_offset_seconds)

    # Save to storage
    storage_dir = Path(settings.STORAGE_ROOT) / "audio-responses"
    storage_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_uuid.uuid4()}_{timestamp}.mp3"
    file_path = storage_dir / filename
    file_path.write_bytes(final_audio)

    return {
        "audio_path": str(file_path),
        "duration_seconds": total_duration,
        "chapter_timestamps": chapter_timestamps,
        "voice_used": voice_id,
        "narration_style": narration_style,
    }
