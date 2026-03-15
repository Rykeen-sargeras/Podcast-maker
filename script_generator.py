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
        api = YouTubeTranscriptApi()
        result = api.fetch(video_id, languages=["en"])
        full_text = " ".join(snippet.text for snippet in result)

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

    except Exception as e:
        error_msg = str(e)
        if "disabled" in error_msg.lower():
            error_msg = "Transcripts are disabled for this video"
        elif "unavailable" in error_msg.lower() or "not found" in error_msg.lower():
            error_msg = "Video is unavailable or not found"
        else:
            error_msg = f"Transcript error: {error_msg}"
        return {"title": "", "text": "", "source": url, "error": error_msg}


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
    duration_min: int,
    duration_max: int,
    custom_speakers: Optional[list[str]],
    additional_instructions: str,
    topic: str,
) -> str:
    """Build the prompt from collected sources and settings."""

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

    # Word count math: ~150 words per minute of spoken audio
    min_words = duration_min * 150
    max_words = int(duration_max * 150 * 1.10)  # allow 10% over the max
    target_words = int((duration_min + duration_max) / 2 * 150)  # aim for midpoint

    prompt = f"""You are a professional podcast script writer. Write a complete, ready-to-read podcast script.

TOPIC: {topic}

PODCAST STYLE: {style_info['description']}
TONE: {style_info['tone']}
SPEAKERS: {speakers_str}

LENGTH REQUIREMENTS (THIS IS CRITICAL — YOU MUST FOLLOW THESE):
- MINIMUM: {min_words} words ({duration_min} minutes). The script MUST be at least this long. A script shorter than this is a failure.
- TARGET: {target_words} words (~{(duration_min + duration_max) // 2} minutes). Aim for this length.
- MAXIMUM: {max_words} words (~{duration_max} minutes + 10% buffer). Do not exceed this.
- At ~150 words per minute of speech, {min_words} words = {duration_min} minutes, {max_words} words = ~{round(max_words / 150)} minutes.
- COUNT YOUR WORDS. If you are under {min_words} words, you MUST add more content. Go deeper into the topics, add more examples, more back-and-forth, more anecdotes. Do NOT wrap up early.

RESEARCH MATERIAL:
{sources_block}

INSTRUCTIONS:
- Write the script in this EXACT format — every line must start with the speaker name followed by a colon:
  Speaker Name: Their dialogue here.
- Make the conversation feel NATURAL — include filler words occasionally, reactions like "Right", "Exactly", "Hmm interesting", interruptions, and conversational flow
- Start with an engaging cold open or hook
- Cover ALL key points from the source material thoroughly — don't skip or summarize, go deep
- When you have lots of source material, discuss each major point in detail with examples and opinions
- Include tangents, personal anecdotes, analogies, and "what if" scenarios to fill out the conversation naturally
- End with a memorable closing / call to action
- Do NOT include stage directions, sound effects, or anything in brackets or parentheses
- Do NOT include episode numbers, timestamps, or metadata
- Each speaker line should be 1-3 sentences (natural speech length)
- The script should flow naturally as if real people are talking
- REMEMBER: The script MUST be at least {min_words} words. Write a LONG, thorough, detailed script. More content is better than less.
{f"- Additional instructions: {additional_instructions}" if additional_instructions else ""}

Write ONLY the script. No preamble, no notes, no explanations before or after. Start directly with the first speaker line."""

    return prompt


# ─── Multi-Provider Script Generation ────────────────────────────────────────

def _call_gemini(api_key: str, prompt: str) -> str:
    """Call Google Gemini API."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=8192,
            temperature=0.85,
        ),
    )
    return response.text.strip()


def _call_openrouter(api_key: str, prompt: str, model_id: str = "deepseek/deepseek-chat-v3-0324:free") -> str:
    """Call OpenRouter API (OpenAI-compatible). Works with free models."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://podforge.app",
        "X-Title": "PodForge",
    }
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "temperature": 0.85,
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_groq(api_key: str, prompt: str) -> str:
    """Call Groq API (fast, free tier)."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "temperature": 0.85,
    }
    resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_script(
    api_key: str,
    sources: list[dict],
    style: str = "interview",
    duration_min: int = 5,
    duration_max: int = 15,
    custom_speakers: Optional[list[str]] = None,
    additional_instructions: str = "",
    topic: str = "",
    provider: str = "gemini",
) -> dict:
    """
    Generate a podcast script using the selected provider.
    Providers: gemini, openrouter, groq
    Returns {"script": str, "error": str|None, "speakers_found": list, "word_count": int, "est_minutes": float}
    """
    if not api_key:
        return {"script": "", "error": "No API key provided", "speakers_found": []}

    try:
        prompt = build_prompt(
            sources=sources,
            style=style,
            duration_min=duration_min,
            duration_max=duration_max,
            custom_speakers=custom_speakers,
            additional_instructions=additional_instructions,
            topic=topic,
        )

        if provider == "groq":
            script = _call_groq(api_key, prompt)
        elif provider == "openrouter":
            script = _call_openrouter(api_key, prompt)
        else:  # gemini (default)
            script = _call_gemini(api_key, prompt)

        # Extract speakers found in the generated script
        speakers = list(dict.fromkeys(
            re.findall(r"^([A-Za-z0-9_ .'-]+?):", script, re.MULTILINE)
        ))

        # Calculate word count and estimated duration
        word_count = len(script.split())
        est_minutes = round(word_count / 150, 1)
        min_words = duration_min * 150
        max_words = int(duration_max * 150 * 1.10)

        warning = None
        if word_count < min_words:
            warning = f"Script is ~{est_minutes} min ({word_count} words) — under the {duration_min} min minimum ({min_words} words). Try regenerating or adding more source material."
        elif word_count > max_words:
            warning = f"Script is ~{est_minutes} min ({word_count} words) — over the {duration_max} min maximum. You may want to trim it."

        return {
            "script": script,
            "error": None,
            "speakers_found": speakers,
            "word_count": word_count,
            "est_minutes": est_minutes,
            "warning": warning,
        }

    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "401" in error_msg or "Unauthorized" in error_msg:
            error_msg = f"Invalid API key for {provider}."
        elif "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
            error_msg = f"Rate limit hit on {provider}. Wait a moment or switch providers."
        elif "safety" in error_msg.lower():
            error_msg = "Content blocked by safety filter. Try adjusting your topic."
        else:
            error_msg = f"{provider} error: {error_msg}"
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
