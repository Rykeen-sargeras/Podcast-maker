"""
PodForge Script Generator
Ingests: YouTube transcripts, web articles, PDFs, raw text
Outputs: Formatted podcast scripts via Google Gemini (free tier)
"""

import os
import re
import json
import traceback
from typing import Optional

import requests
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


# ─── YouTube Transcript Extraction ───────────────────────────────────────────

def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",  # bare ID
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def get_youtube_transcript(url: str) -> dict:
    """
    Fetch transcript from a YouTube video.
    Returns {"title": str, "text": str, "source": str, "error": str|None}
    """
    video_id = extract_video_id(url)
    if not video_id:
        return {"title": "", "text": "", "source": url, "error": "Could not parse YouTube URL"}

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer manual English, then auto-generated English, then any
        transcript = None
        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
        except NoTranscriptFound:
            try:
                transcript = transcript_list.find_generated_transcript(["en"])
            except NoTranscriptFound:
                # Get whatever is available
                for t in transcript_list:
                    transcript = t
                    break

        if transcript is None:
            return {"title": "", "text": "", "source": url, "error": "No transcript available"}

        entries = transcript.fetch()
        full_text = " ".join(entry.get("text", entry.get("value", "")) if isinstance(entry, dict) else str(entry) for entry in entries)

        # Try to get video title via oembed (no API key needed)
        title = f"YouTube Video ({video_id})"
        try:
            oembed = requests.get(
                f"https://www.youtube.com/oembed?url=https://youtube.com/watch?v={video_id}&format=json",
                timeout=5,
            )
            if oembed.ok:
                title = oembed.json().get("title", title)
        except:
            pass

        return {"title": title, "text": full_text, "source": url, "error": None}

    except TranscriptsDisabled:
        return {"title": "", "text": "", "source": url, "error": "Transcripts are disabled for this video"}
    except VideoUnavailable:
        return {"title": "", "text": "", "source": url, "error": "Video is unavailable"}
    except Exception as e:
        return {"title": "", "text": "", "source": url, "error": f"Transcript error: {str(e)}"}


# ─── Web Article Extraction ──────────────────────────────────────────────────

def scrape_article(url: str) -> dict:
    """
    Extract main text content from a web article.
    Returns {"title": str, "text": str, "source": str, "error": str|None}
    """
    if trafilatura is None:
        return {"title": "", "text": "", "source": url, "error": "trafilatura not installed"}

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return {"title": "", "text": "", "source": url, "error": "Could not download page"}

        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        metadata = trafilatura.extract(downloaded, output_format="json")

        title = url
        if metadata:
            try:
                meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                title = meta_dict.get("title", url) or url
            except:
                pass

        if not text:
            return {"title": title, "text": "", "source": url, "error": "Could not extract text from page"}

        return {"title": title, "text": text, "source": url, "error": None}

    except Exception as e:
        return {"title": "", "text": "", "source": url, "error": f"Scrape error: {str(e)}"}


# ─── PDF Extraction ──────────────────────────────────────────────────────────

def extract_pdf_text(file_path: str) -> dict:
    """
    Extract text from a PDF file.
    Returns {"title": str, "text": str, "source": str, "error": str|None}
    """
    if PdfReader is None:
        return {"title": "", "text": "", "source": file_path, "error": "PyPDF2 not installed"}

    try:
        reader = PdfReader(file_path)
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)

        full_text = "\n".join(pages_text)
        if not full_text.strip():
            return {"title": os.path.basename(file_path), "text": "", "source": file_path,
                    "error": "PDF appears to be image-based (no extractable text)"}

        title = os.path.basename(file_path)
        # Try to get title from metadata
        if reader.metadata and reader.metadata.title:
            title = reader.metadata.title

        return {"title": title, "text": full_text, "source": file_path, "error": None}

    except Exception as e:
        return {"title": "", "text": "", "source": file_path, "error": f"PDF error: {str(e)}"}


# ─── Gemini Script Generation ────────────────────────────────────────────────

STYLE_PROMPTS = {
    "interview": {
        "speakers": "Host, Guest",
        "description": "a natural conversational interview podcast where the host asks thoughtful questions and the guest shares insights, stories, and expertise",
        "tone": "curious, engaging, conversational",
    },
    "debate": {
        "speakers": "Moderator, Speaker A, Speaker B",
        "description": "a structured but lively debate where two speakers present opposing viewpoints with a moderator guiding the discussion",
        "tone": "passionate, respectful disagreement, analytical",
    },
    "panel": {
        "speakers": "Host, Panelist 1, Panelist 2, Panelist 3",
        "description": "a roundtable panel discussion where multiple experts share different perspectives on the topic",
        "tone": "collaborative, insightful, varied perspectives",
    },
    "solo": {
        "speakers": "Host",
        "description": "a solo commentary podcast where one person shares their thoughts, analysis, and opinions directly with the audience",
        "tone": "personal, reflective, direct, engaging",
    },
    "storytelling": {
        "speakers": "Narrator",
        "description": "a narrative storytelling podcast that weaves facts and information into a compelling story with vivid descriptions and dramatic pacing",
        "tone": "immersive, dramatic, descriptive, captivating",
    },
}


def build_prompt(
    sources: list[dict],
    style: str,
    duration_minutes: int,
    custom_speakers: Optional[list[str]],
    additional_instructions: str,
    topic: str,
) -> str:
    """Build the Gemini prompt from collected sources and settings."""

    style_info = STYLE_PROMPTS.get(style, STYLE_PROMPTS["interview"])

    # Determine speakers
    if custom_speakers and len(custom_speakers) > 0:
        speakers_str = ", ".join(custom_speakers)
    else:
        speakers_str = style_info["speakers"]

    # Build source material section
    source_sections = []
    for i, src in enumerate(sources, 1):
        if src.get("text"):
            # Truncate very long sources to ~4000 chars each to stay within context limits
            text = src["text"][:4000]
            if len(src["text"]) > 4000:
                text += "\n[... truncated for length ...]"
            source_sections.append(
                f"--- SOURCE {i}: {src.get('title', 'Untitled')} ---\n"
                f"(From: {src.get('source', 'unknown')})\n\n{text}"
            )

    sources_block = "\n\n".join(source_sections) if source_sections else "(No source material provided — generate based on the topic alone.)"

    # Estimate word count (~150 words per minute of spoken audio)
    target_words = duration_minutes * 150

    prompt = f"""You are a professional podcast script writer. Write a complete, ready-to-read podcast script.

TOPIC: {topic}

PODCAST STYLE: {style_info['description']}
TONE: {style_info['tone']}
SPEAKERS: {speakers_str}
TARGET LENGTH: ~{target_words} words ({duration_minutes} minutes of spoken audio)

RESEARCH MATERIAL:
{sources_block}

INSTRUCTIONS:
- Write the script in this EXACT format — every line must start with the speaker name followed by a colon:
  Speaker Name: Their dialogue here.
- Make the conversation feel NATURAL — include filler words occasionally, reactions like "Right", "Exactly", "Hmm interesting", interruptions, and conversational flow
- Start with an engaging cold open or hook
- Cover the key points from the source material accurately
- End with a memorable closing / call to action
- Do NOT include stage directions, sound effects, or anything in brackets or parentheses
- Do NOT include episode numbers, timestamps, or metadata
- Each speaker line should be 1-3 sentences (natural speech length)
- The script should flow naturally as if real people are talking
{f"- Additional instructions: {additional_instructions}" if additional_instructions else ""}

Write ONLY the script. No preamble, no notes, no explanations before or after. Start directly with the first speaker line."""

    return prompt


def generate_script(
    api_key: str,
    sources: list[dict],
    style: str = "interview",
    duration_minutes: int = 10,
    custom_speakers: Optional[list[str]] = None,
    additional_instructions: str = "",
    topic: str = "",
) -> dict:
    """
    Generate a podcast script using Google Gemini.
    Returns {"script": str, "error": str|None, "speakers_found": list}
    """
    if not api_key:
        return {"script": "", "error": "No Gemini API key provided", "speakers_found": []}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = build_prompt(
            sources=sources,
            style=style,
            duration_minutes=duration_minutes,
            custom_speakers=custom_speakers,
            additional_instructions=additional_instructions,
            topic=topic,
        )

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=8192,
                temperature=0.85,
            ),
        )

        script = response.text.strip()

        # Extract speakers found in the generated script
        speakers = list(dict.fromkeys(
            re.findall(r"^([A-Za-z0-9_ .'-]+?):", script, re.MULTILINE)
        ))

        return {"script": script, "error": None, "speakers_found": speakers}

    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "401" in error_msg:
            error_msg = "Invalid Gemini API key. Get a free one at https://aistudio.google.com/apikey"
        elif "429" in error_msg or "quota" in error_msg.lower():
            error_msg = "Rate limit hit. Free tier allows 15 requests/minute. Wait a moment and try again."
        elif "safety" in error_msg.lower():
            error_msg = "Content was blocked by Gemini's safety filter. Try adjusting your topic or instructions."
        return {"script": "", "error": error_msg, "speakers_found": []}


# ─── Convenience: Process all inputs at once ─────────────────────────────────

def ingest_all_sources(
    youtube_urls: list[str] = None,
    web_urls: list[str] = None,
    raw_texts: list[dict] = None,
    pdf_paths: list[str] = None,
) -> tuple[list[dict], list[str]]:
    """
    Process all input sources and return (sources, warnings).
    Each source: {"title": str, "text": str, "source": str, "error": str|None}
    """
    sources = []
    warnings = []

    # YouTube
    for url in (youtube_urls or []):
        url = url.strip()
        if not url:
            continue
        result = get_youtube_transcript(url)
        if result["error"]:
            warnings.append(f"YouTube ({url}): {result['error']}")
        if result["text"]:
            sources.append(result)

    # Web articles
    for url in (web_urls or []):
        url = url.strip()
        if not url:
            continue
        result = scrape_article(url)
        if result["error"]:
            warnings.append(f"Article ({url}): {result['error']}")
        if result["text"]:
            sources.append(result)

    # Raw text
    for item in (raw_texts or []):
        if isinstance(item, str):
            item = {"title": "Pasted Notes", "text": item}
        if item.get("text", "").strip():
            sources.append({
                "title": item.get("title", "Pasted Notes"),
                "text": item["text"],
                "source": "user input",
                "error": None,
            })

    # PDFs
    for path in (pdf_paths or []):
        result = extract_pdf_text(path)
        if result["error"]:
            warnings.append(f"PDF ({path}): {result['error']}")
        if result["text"]:
            sources.append(result)

    return sources, warnings
